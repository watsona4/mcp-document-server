#!/usr/bin/env python3
"""
Ephemeral Vultr GPU automation for Whisper transcription.

Scans for audio files without transcripts, spins up a Vultr GPU instance,
transcribes all pending files, then destroys the VM.

Writes .vtt files directly to the recordings directory — the existing
MCP document server's audio scanner detects these and skips re-transcription.

Usage:
    python3 vultr-whisper.py              # Process all pending files
    python3 vultr-whisper.py --dry-run    # Show what would be done
    python3 vultr-whisper.py --cleanup    # Destroy orphaned VMs only

Zero external dependencies (stdlib only).
"""

import argparse
import fcntl
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("vultr-whisper")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS = {".m4a", ".mp4", ".wav", ".mp3", ".ogg", ".flac", ".wma", ".aac", ".opus", ".webm"}


def load_config() -> dict:
    """Load configuration from .env.vultr file and environment variables."""
    config = {}

    # Load .env.vultr if present (next to this script)
    env_file = Path(__file__).parent / ".env.vultr"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip().strip("'\"")

    # Environment variables override the file
    for key in (
        "VULTR_API_KEY", "HF_TOKEN", "RECORDINGS_DIR", "SPEAKERS_DIR",
        "VULTR_REGION", "VULTR_PLAN", "VULTR_OS", "VULTR_SNAPSHOT",
        "WHISPER_MODEL", "MAX_RUNTIME_MINUTES", "WHISPER_PORT", "SSH_KEY_PATH",
    ):
        val = os.environ.get(key)
        if val:
            config[key] = val

    # Apply defaults
    config.setdefault("RECORDINGS_DIR", "/mnt/gdrive/Recordings")
    config.setdefault("SPEAKERS_DIR", "speakers")
    config.setdefault("VULTR_REGION", "ewr")
    config.setdefault("VULTR_PLAN", "vcg-a16-6c-64g-16vram")
    config.setdefault("VULTR_OS", "1743")  # Ubuntu 22.04
    config.setdefault("WHISPER_MODEL", "large-v3")
    config.setdefault("MAX_RUNTIME_MINUTES", "120")
    config.setdefault("WHISPER_PORT", "9000")
    config.setdefault("SSH_KEY_PATH", str(Path.home() / ".ssh" / "vultr_whisper_ed25519"))

    return config


# ---------------------------------------------------------------------------
# File normalization — rename recordings to YYYY-MM-DD_HHMM_meeting format
# ---------------------------------------------------------------------------

# Matches "Feb 17 at 9-45 AM", "Feb 6 at 1-44 PM", etc.
_OLD_NAME_RE = re.compile(
    r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+'
    r'(\d{1,2})\s+at\s+(\d{1,2})-(\d{2})\s+(AM|PM)$'
)

_MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


def normalize_filename(path: Path) -> Path:
    """Rename 'Feb 17 at 9-45 AM.m4a' → '2026-02-17_0945_meeting.m4a'.

    Derives the year from the file's modification time.
    Returns the new path, or the original path if no rename was needed.
    """
    m = _OLD_NAME_RE.match(path.stem)
    if not m:
        return path  # already normalized or unrecognized

    month_str, day_str, hour_str, minute_str, ampm = m.groups()

    # Year from mtime (not in the filename)
    year = datetime.fromtimestamp(path.stat().st_mtime).year

    month = _MONTH_MAP[month_str]
    day = int(day_str)
    hour = int(hour_str)
    minute = int(minute_str)
    if ampm == 'PM' and hour != 12:
        hour += 12
    elif ampm == 'AM' and hour == 12:
        hour = 0

    new_stem = f"{year}-{month:02d}-{day:02d}_{hour:02d}{minute:02d}_meeting"
    new_path = path.with_stem(new_stem)

    if new_path.exists():
        log.warning("Target already exists, skipping rename: %s → %s", path.name, new_path.name)
        return path

    path.rename(new_path)
    log.info("Renamed: %s → %s", path.name, new_path.name)
    return new_path


def normalize_recordings(config: dict) -> int:
    """Rename all old-format recordings and transcripts to normalized names.

    Returns the number of files renamed.
    """
    recordings = Path(config["RECORDINGS_DIR"])
    if not recordings.is_dir():
        return 0

    # Collect matching files first (don't modify dir while iterating)
    to_rename = [
        f for f in recordings.iterdir()
        if f.is_file() and _OLD_NAME_RE.match(f.stem)
    ]

    if not to_rename:
        return 0

    renamed = 0
    for f in sorted(to_rename, key=lambda p: p.name):
        new = normalize_filename(f)
        if new != f:
            renamed += 1

    if renamed:
        log.info("Normalized %d file name(s)", renamed)
    return renamed


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def find_pending_files(config: dict) -> list[Path]:
    """Return audio files that have no matching .vtt transcript."""
    recordings = Path(config["RECORDINGS_DIR"])
    if not recordings.is_dir():
        log.warning("Recordings directory does not exist: %s", recordings)
        return []

    speakers_dir = recordings / config["SPEAKERS_DIR"]
    pending = []

    for f in recordings.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        # Skip speaker enrollment clips
        try:
            f.relative_to(speakers_dir)
            continue
        except ValueError:
            pass
        # Check for existing transcript
        vtt = f.with_suffix(".vtt")
        if not vtt.exists():
            pending.append(f)

    # Sort oldest first so we process in chronological order
    pending.sort(key=lambda p: p.stat().st_mtime)
    return pending


# ---------------------------------------------------------------------------
# Vultr API helpers
# ---------------------------------------------------------------------------

def vultr_api(method: str, path: str, api_key: str, data: dict | None = None) -> dict | None:
    """Make a Vultr API v2 request. Returns parsed JSON or None for 204."""
    url = f"https://api.vultr.com/v2{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = json.dumps(data).encode() if data else None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 204:
                return None
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        log.error("Vultr API %s %s -> %d: %s", method, path, e.code, error_body)
        raise


# ---------------------------------------------------------------------------
# SSH key management
# ---------------------------------------------------------------------------

def ensure_ssh_key(config: dict) -> str:
    """Generate a dedicated SSH key if missing, register it with Vultr.

    Returns the Vultr SSH key ID.
    """
    key_path = Path(config["SSH_KEY_PATH"])
    pub_path = key_path.with_suffix(".pub")
    api_key = config["VULTR_API_KEY"]

    # Generate key if it doesn't exist
    if not key_path.exists():
        log.info("Generating dedicated SSH key: %s", key_path)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-C", "vultr-whisper-ephemeral"],
            check=True, capture_output=True,
        )
        log.info("SSH key generated")

    pub_key = pub_path.read_text().strip()

    # Check if already registered with Vultr
    resp = vultr_api("GET", "/ssh-keys", api_key)
    for key in resp.get("ssh_keys", []):
        if key.get("ssh_key", "").strip() == pub_key:
            log.info("SSH key already registered with Vultr: %s", key["id"])
            return key["id"]

    # Register new key
    log.info("Registering SSH key with Vultr...")
    resp = vultr_api("POST", "/ssh-keys", api_key, {
        "name": "vultr-whisper-ephemeral",
        "ssh_key": pub_key,
    })
    key_id = resp["ssh_key"]["id"]
    log.info("SSH key registered: %s", key_id)
    return key_id


# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------

def _build_cloud_init(config: dict) -> str:
    """Build the cloud-init user-data script."""
    hf_token = config.get("HF_TOKEN", "")
    model = config["WHISPER_MODEL"]
    port = config["WHISPER_PORT"]

    hf_env = f"-e HF_TOKEN={hf_token}" if hf_token else ""

    return f"""#!/bin/bash
set -euo pipefail
exec > /var/log/whisper-init.log 2>&1

# Install NVIDIA container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Start whisper container with GPU
docker run -d --gpus all -p {port}:{port} -e ASR_MODEL={model} -e ASR_ENGINE=whisperx {hf_env} --name whisper onerahmet/openai-whisper-asr-webservice:latest-gpu

# Signal completion
touch /tmp/cloud-init-done
"""


def create_gpu_instance(config: dict, ssh_key_id: str) -> str:
    """Create a Vultr GPU instance. Returns the instance ID."""
    api_key = config["VULTR_API_KEY"]
    snapshot_id = config.get("VULTR_SNAPSHOT")

    instance_data = {
        "region": config["VULTR_REGION"],
        "plan": config["VULTR_PLAN"],
        "label": "whisper-ephemeral",
        "tag": "whisper-ephemeral",
        "tags": ["whisper-ephemeral"],
        "sshkey_id": [ssh_key_id],
        "backups": "disabled",
        "activation_email": False,
    }

    if snapshot_id:
        # Boot from pre-built snapshot (fast — no cloud-init needed)
        log.info("Creating instance from snapshot %s (plan=%s, region=%s)...",
                 snapshot_id, config["VULTR_PLAN"], config["VULTR_REGION"])
        instance_data["snapshot_id"] = snapshot_id
    else:
        # Fresh OS + cloud-init setup (slow — installs toolkit, pulls image)
        import base64
        user_data = _build_cloud_init(config)
        user_data_b64 = base64.b64encode(user_data.encode()).decode()
        log.info("Creating instance from OS (plan=%s, region=%s)...",
                 config["VULTR_PLAN"], config["VULTR_REGION"])
        instance_data["os_id"] = int(config["VULTR_OS"])
        instance_data["user_data"] = user_data_b64

    resp = vultr_api("POST", "/instances", api_key, instance_data)
    instance_id = resp["instance"]["id"]
    log.info("Instance created: %s", instance_id)
    return instance_id


def get_instance_info(config: dict, instance_id: str) -> dict:
    """Get instance details from Vultr API."""
    return vultr_api("GET", f"/instances/{instance_id}", config["VULTR_API_KEY"])["instance"]


def wait_for_instance(config: dict, instance_id: str, timeout: int = 900) -> str:
    """Wait for instance to be active and cloud-init to complete.

    Returns the instance's main IP address.
    """
    api_key = config["VULTR_API_KEY"]
    key_path = config["SSH_KEY_PATH"]
    deadline = time.time() + timeout
    start_sent = False

    # Phase 1: wait for Vultr status=active, power=running, and an IP
    log.info("Waiting for instance to become active...")
    ip_addr = None
    while time.time() < deadline:
        info = get_instance_info(config, instance_id)
        status = info.get("status")
        power = info.get("power_status")
        server = info.get("server_status")
        ip_addr = info.get("main_ip")

        if status == "active" and power == "running" and ip_addr and ip_addr != "0.0.0.0":
            log.info("Instance active at %s", ip_addr)
            break

        # GPU instances can end up active+stopped+ok — send explicit start
        if status == "active" and power == "stopped" and server == "ok" and not start_sent:
            log.info("Instance provisioned but stopped — sending start command...")
            try:
                vultr_api("POST", f"/instances/{instance_id}/start", api_key)
            except Exception as e:
                log.warning("Start command returned error (may still work): %s", e)
            start_sent = True

        log.info("Instance: status=%s power=%s server=%s ip=%s", status, power, server, ip_addr)
        time.sleep(15)
    else:
        raise TimeoutError(f"Instance {instance_id} did not become active within {timeout}s")

    # Phase 2: wait for SSH to respond
    log.info("Waiting for SSH on %s...", ip_addr)
    ssh_opts = _ssh_opts(key_path)
    while time.time() < deadline:
        try:
            result = subprocess.run(
                ["ssh", *ssh_opts, f"root@{ip_addr}", "echo ready"],
                capture_output=True, text=True, timeout=20,
            )
            if result.returncode == 0 and "ready" in result.stdout:
                log.info("SSH is ready")
                break
        except subprocess.TimeoutExpired:
            pass
        time.sleep(10)
    else:
        raise TimeoutError(f"SSH did not become available on {ip_addr} within {timeout}s")

    # Phase 3: wait for cloud-init (fresh OS) or Docker (snapshot) to finish
    if config.get("VULTR_SNAPSHOT"):
        # Snapshot boot — just verify Docker and whisper container are running
        log.info("Snapshot boot — waiting for whisper container to start...")
        while time.time() < deadline:
            try:
                result = subprocess.run(
                    ["ssh", *ssh_opts, f"root@{ip_addr}",
                     "docker ps --filter name=whisper --format '{{.Status}}'"],
                    capture_output=True, text=True, timeout=20,
                )
                if result.returncode == 0 and "Up" in result.stdout:
                    log.info("Whisper container is running")
                    return ip_addr
                # Container might need a restart after snapshot boot
                if result.returncode == 0 and not result.stdout.strip():
                    log.info("Whisper container not running, starting it...")
                    subprocess.run(
                        ["ssh", *ssh_opts, f"root@{ip_addr}", "docker start whisper"],
                        capture_output=True, timeout=30,
                    )
            except subprocess.TimeoutExpired:
                pass
            time.sleep(10)
    else:
        log.info("Waiting for cloud-init to complete...")
        while time.time() < deadline:
            try:
                result = subprocess.run(
                    ["ssh", *ssh_opts, f"root@{ip_addr}", "test -f /tmp/cloud-init-done && echo done"],
                    capture_output=True, text=True, timeout=20,
                )
                if result.returncode == 0 and "done" in result.stdout:
                    log.info("Cloud-init complete")
                    return ip_addr
            except subprocess.TimeoutExpired:
                pass
            time.sleep(15)

    raise TimeoutError(f"Instance setup did not complete on {ip_addr} within {timeout}s")


def wait_for_whisper(ip_addr: str, config: dict, timeout: int = 600) -> None:
    """Poll the whisper health endpoint until the model is loaded and ready."""
    port = config["WHISPER_PORT"]
    url = f"http://{ip_addr}:{port}/docs"
    deadline = time.time() + timeout

    log.info("Waiting for Whisper service at %s:%s...", ip_addr, port)
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    log.info("Whisper service is ready")
                    return
        except Exception:
            pass
        time.sleep(15)

    raise TimeoutError(f"Whisper service did not become ready on {ip_addr}:{port} within {timeout}s")


def destroy_instance(config: dict, instance_id: str) -> None:
    """Delete a Vultr instance."""
    log.info("Destroying instance %s...", instance_id)
    try:
        vultr_api("DELETE", f"/instances/{instance_id}", config["VULTR_API_KEY"])
        log.info("Instance %s destroyed", instance_id)
    except Exception as e:
        log.error("Failed to destroy instance %s: %s", instance_id, e)


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------

def cleanup_orphans(config: dict) -> None:
    """Find and destroy VMs tagged whisper-ephemeral older than MAX_RUNTIME."""
    api_key = config["VULTR_API_KEY"]
    max_minutes = int(config["MAX_RUNTIME_MINUTES"])

    try:
        resp = vultr_api("GET", "/instances", api_key)
    except Exception as e:
        log.warning("Could not list instances for orphan cleanup: %s", e)
        return

    now = time.time()
    for inst in resp.get("instances", []):
        tags = inst.get("tags", [])
        label = inst.get("label", "")

        if "whisper-ephemeral" not in tags and label != "whisper-ephemeral":
            continue

        # Parse creation date
        date_created = inst.get("date_created", "")
        if not date_created:
            continue

        try:
            # Vultr dates look like "2024-01-15T10:30:00+00:00"
            from datetime import datetime, timezone
            created = datetime.fromisoformat(date_created).timestamp()
            age_minutes = (now - created) / 60

            if age_minutes > max_minutes:
                log.warning(
                    "Destroying orphaned instance %s (age=%.0f min, max=%d min)",
                    inst["id"], age_minutes, max_minutes,
                )
                destroy_instance(config, inst["id"])
        except Exception as e:
            log.warning("Error checking orphan %s: %s", inst.get("id"), e)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _ssh_opts(key_path: str) -> list[str]:
    """Common SSH/SCP options for non-interactive use."""
    return [
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=15",
    ]


def transcribe_file(audio_path: Path, ip_addr: str, config: dict) -> bool:
    """Upload, transcribe, download VTT for a single file.

    Returns True on success.
    """
    key_path = config["SSH_KEY_PATH"]
    port = config["WHISPER_PORT"]
    ssh_opts = _ssh_opts(key_path)
    remote_audio = "/tmp/audio" + audio_path.suffix
    local_vtt = audio_path.with_suffix(".vtt")

    log.info("Transcribing: %s", audio_path.name)

    for attempt in range(2):
        if attempt > 0:
            log.info("Retrying %s (attempt %d)...", audio_path.name, attempt + 1)
            time.sleep(5)

        try:
            # 1. Upload audio file
            log.info("  Uploading %s (%.1f MB)...", audio_path.name, audio_path.stat().st_size / 1024 / 1024)
            result = subprocess.run(
                ["scp", *ssh_opts, str(audio_path), f"root@{ip_addr}:{remote_audio}"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                log.error("  SCP upload failed: %s", result.stderr)
                continue

            # 2. Run transcription via curl inside the VM
            whisper_url = (
                f"http://localhost:{port}/asr"
                f"?task=transcribe&output=vtt&language=en&diarize=true"
            )
            curl_cmd = f"curl -s -X POST '{whisper_url}' -F 'audio_file=@{remote_audio}'"

            log.info("  Running transcription...")
            result = subprocess.run(
                ["ssh", *ssh_opts, f"root@{ip_addr}", curl_cmd],
                capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                log.error("  SSH curl failed: %s", result.stderr)
                continue

            vtt_content = result.stdout

            # 3. Validate VTT
            if not validate_vtt(vtt_content):
                log.error("  VTT validation failed (no timestamps or too small)")
                continue

            # 4. Write VTT to local filesystem
            with open(local_vtt, "w", encoding="utf-8") as f:
                f.write(vtt_content)
            log.info("  Saved: %s (%d bytes)", local_vtt.name, len(vtt_content.encode()))

            # 5. Cleanup remote audio
            subprocess.run(
                ["ssh", *ssh_opts, f"root@{ip_addr}", f"rm -f {remote_audio}"],
                capture_output=True, timeout=30,
            )

            return True

        except subprocess.TimeoutExpired:
            log.error("  Timeout during transcription of %s", audio_path.name)
            continue
        except Exception as e:
            log.error("  Error transcribing %s: %s", audio_path.name, e)
            continue

    log.error("Failed to transcribe %s after retries", audio_path.name)
    return False


def validate_vtt(content: str) -> bool:
    """Check that VTT content has timestamp markers and non-trivial size."""
    if not content or len(content) < 50:
        return False
    if "-->" not in content:
        return False
    return True


# ---------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------

LOCK_PATH = Path("/tmp/vultr-whisper.lock")


def acquire_lock() -> 'int | None':
    """Acquire a lock file. Returns the file descriptor, or None if already locked."""
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_WRONLY, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Write our PID
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except OSError:
        # Check if the locking PID is still alive (stale lock)
        try:
            with open(LOCK_PATH) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # Check if process exists
            log.warning("Another instance is running (PID %d), exiting", pid)
            return None
        except (ValueError, ProcessLookupError, FileNotFoundError):
            # Stale lock — remove and retry
            log.info("Removing stale lock file")
            LOCK_PATH.unlink(missing_ok=True)
            return acquire_lock()


def release_lock(fd: int) -> None:
    """Release the lock file."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Ephemeral Vultr GPU Whisper transcription")
    parser.add_argument("--dry-run", action="store_true", help="Show pending files without processing")
    parser.add_argument("--cleanup", action="store_true", help="Only destroy orphaned VMs")
    args = parser.parse_args()

    config = load_config()

    # Validate required config
    if not config.get("VULTR_API_KEY"):
        log.error("VULTR_API_KEY is required. Set it in .env.vultr or environment.")
        return 1

    log.info("=" * 60)
    log.info("Vultr Whisper - Ephemeral GPU Transcription")
    log.info("=" * 60)
    log.info("Recordings: %s", config["RECORDINGS_DIR"])
    log.info("Model: %s", config["WHISPER_MODEL"])
    log.info("Plan: %s in %s", config["VULTR_PLAN"], config["VULTR_REGION"])
    if config.get("VULTR_SNAPSHOT"):
        log.info("Snapshot: %s (fast boot)", config["VULTR_SNAPSHOT"])
    else:
        log.info("Boot: fresh OS + cloud-init (slow)")
    log.info("Max runtime: %s min", config["MAX_RUNTIME_MINUTES"])

    # Always clean up orphans first
    cleanup_orphans(config)

    if args.cleanup:
        log.info("Cleanup-only mode, exiting")
        return 0

    # Normalize filenames before scanning
    normalize_recordings(config)

    # Find pending files
    pending = find_pending_files(config)
    if not pending:
        log.info("No pending files to transcribe. Exiting.")
        return 0

    log.info("Found %d pending file(s):", len(pending))
    for f in pending:
        log.info("  - %s (%.1f MB)", f.name, f.stat().st_size / 1024 / 1024)

    if args.dry_run:
        log.info("Dry-run mode, exiting without creating VM")
        return 0

    # Acquire lock
    lock_fd = acquire_lock()
    if lock_fd is None:
        return 0  # Another instance running, not an error

    instance_id = None
    start_time = time.time()

    # Safety: always destroy VM on signal
    def _signal_handler(signum, frame):
        log.warning("Received signal %d, cleaning up...", signum)
        if instance_id:
            try:
                destroy_instance(config, instance_id)
            except Exception:
                pass
        release_lock(lock_fd)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        # Ensure SSH key is ready
        ssh_key_id = ensure_ssh_key(config)

        # Create GPU instance
        instance_id = create_gpu_instance(config, ssh_key_id)

        # Wait for it to be ready
        ip_addr = wait_for_instance(config, instance_id, timeout=1800)
        wait_for_whisper(ip_addr, config, timeout=600)

        setup_elapsed = time.time() - start_time
        log.info("VM ready in %.0fs. Starting batch transcription...", setup_elapsed)

        # Transcribe all pending files
        max_runtime = int(config["MAX_RUNTIME_MINUTES"])
        success = 0
        failed = 0
        skipped = 0

        for i, audio_path in enumerate(pending, 1):
            # Safety: check max runtime
            elapsed_min = (time.time() - start_time) / 60
            if elapsed_min > max_runtime:
                remaining = len(pending) - i + 1
                log.warning("Max runtime exceeded (%.0f > %d min). Skipping %d remaining file(s).",
                            elapsed_min, max_runtime, remaining)
                skipped = remaining
                break

            log.info("[%d/%d] Processing %s", i, len(pending), audio_path.name)
            if transcribe_file(audio_path, ip_addr, config):
                success += 1
            else:
                failed += 1

        total_elapsed = time.time() - start_time
        cost_estimate = (total_elapsed / 3600) * 0.471  # $0.471/hr for A16 16GB VRAM

        log.info("=" * 60)
        log.info("Batch complete: %d succeeded, %d failed, %d skipped", success, failed, skipped)
        log.info("Total time: %.1f min (setup: %.0fs)", total_elapsed / 60, setup_elapsed)
        log.info("Estimated cost: $%.2f", cost_estimate)
        log.info("=" * 60)

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 130
    except Exception as e:
        log.error("Unhandled error: %s", e, exc_info=True)
        return 1
    finally:
        # Always destroy the VM
        if instance_id:
            try:
                destroy_instance(config, instance_id)
            except Exception as e:
                log.error("CRITICAL: Failed to destroy VM %s: %s", instance_id, e)
                log.error("Manual cleanup needed!")

        release_lock(lock_fd)


if __name__ == "__main__":
    sys.exit(main())
