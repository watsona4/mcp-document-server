#!/usr/bin/env python3
"""
Extract speaker audio clips from diarized VTT transcripts.

Parses a VTT file to find speaker segments, shows a summary so you can
identify who's who, then extracts audio clips for each speaker using ffmpeg.

Supports two VTT formats:
  - WhisperX diarized: [SPEAKER_00]: text
  - Teams/WebVTT voice tags: <v Speaker Name>text</v>

Usage:
  python extract-speakers.py <audio_file> [--vtt <vtt_file>] [--outdir <speakers_dir>]

Examples:
  python extract-speakers.py "Feb 12 at 12-56 PM.m4a"
  python extract-speakers.py recording.m4a --vtt recording.vtt --outdir /mnt/gdrive/Recordings/speakers

Run against multiple meetings to build diverse enrollment profiles:
  python extract-speakers.py "Feb 11 at 1-01 PM.m4a"
  python extract-speakers.py "Feb 12 at 12-56 PM.m4a"
  python extract-speakers.py "Feb 13 at 1-14 PM.m4a"
  # Each run appends new clips into speakers/Name/ without overwriting
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_timestamp(ts: str) -> float:
    """Parse VTT timestamp to seconds. Handles HH:MM:SS.mmm and MM:SS.mmm."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(ts)


def format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_vtt(vtt_path: Path) -> list[dict]:
    """Parse VTT file and return list of segments with speaker info."""
    text = vtt_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.strip().split("\n")

    # Patterns
    ts_pattern = re.compile(r"(\d[\d:.]+)\s*-->\s*(\d[\d:.]+)")
    whisperx_pattern = re.compile(r"^\[([^\]]+)\]:\s*(.+)")
    voice_tag_pattern = re.compile(r"<v\s+([^>]+)>(.+?)</v>")

    segments = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp line
        ts_match = ts_pattern.match(line)
        if ts_match:
            start = parse_timestamp(ts_match.group(1))
            end = parse_timestamp(ts_match.group(2))

            # Collect text lines until blank line or next timestamp
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not ts_pattern.match(lines[i].strip()):
                text_lines.append(lines[i].strip())
                i += 1

            full_text = " ".join(text_lines)
            speaker = None

            # Try WhisperX format: [SPEAKER_00]: text
            wx_match = whisperx_pattern.match(full_text)
            if wx_match:
                speaker = wx_match.group(1)
                full_text = wx_match.group(2).strip()

            # Try voice tag format: <v Name>text</v>
            if not speaker:
                vt_match = voice_tag_pattern.search(full_text)
                if vt_match:
                    speaker = vt_match.group(1).strip()
                    full_text = vt_match.group(2).strip()

            if full_text:
                segments.append({
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": full_text,
                })
        else:
            i += 1

    return segments


def summarize_speakers(segments: list[dict]) -> dict[str, dict]:
    """Build per-speaker summary: total time, segment count, sample text."""
    speakers = {}
    for seg in segments:
        spk = seg["speaker"]
        if spk is None:
            continue
        if spk not in speakers:
            speakers[spk] = {
                "total_time": 0.0,
                "segments": [],
                "sample_texts": [],
            }
        duration = seg["end"] - seg["start"]
        speakers[spk]["total_time"] += duration
        speakers[spk]["segments"].append(seg)
        if len(speakers[spk]["sample_texts"]) < 5:
            speakers[spk]["sample_texts"].append(
                f"  [{format_timestamp(seg['start'])}] \"{seg['text'][:80]}\""
            )

    return speakers


def pick_best_segments(
    segments: list[dict],
    target_seconds: float = 30.0,
    min_duration: float = 3.0,
    max_clips: int = 3,
) -> list[dict]:
    """Pick the longest non-overlapping segments up to target total duration.

    Limits to max_clips per meeting to encourage diversity across meetings.
    """
    # Sort by duration descending
    candidates = [s for s in segments if (s["end"] - s["start"]) >= min_duration]
    candidates.sort(key=lambda s: s["end"] - s["start"], reverse=True)

    picked = []
    total = 0.0
    for seg in candidates:
        if total >= target_seconds or len(picked) >= max_clips:
            break
        picked.append(seg)
        total += seg["end"] - seg["start"]

    # If we didn't get enough from long segments, try shorter ones (>=1.5s)
    if total < min_duration:
        shorter = [s for s in segments if (s["end"] - s["start"]) >= 1.5 and s not in picked]
        shorter.sort(key=lambda s: s["end"] - s["start"], reverse=True)
        for seg in shorter:
            if total >= target_seconds or len(picked) >= max_clips:
                break
            picked.append(seg)
            total += seg["end"] - seg["start"]

    # Sort by time for cleaner output
    picked.sort(key=lambda s: s["start"])
    return picked


def get_existing_clips(speaker_dir: Path) -> list[Path]:
    """Return existing clip files in a speaker directory, sorted."""
    if not speaker_dir.exists():
        return []
    return sorted(speaker_dir.glob("*.wav"))


def next_clip_number(speaker_dir: Path) -> int:
    """Find the next available clip number for a speaker directory."""
    existing = get_existing_clips(speaker_dir)
    if not existing:
        return 1
    # Extract numbers from filenames like clip3_feb12.wav -> 3
    numbers = []
    for p in existing:
        match = re.match(r"clip(\d+)", p.stem)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def make_source_tag(audio_path: Path) -> str:
    """Create a short source tag from the audio filename for clip traceability."""
    stem = audio_path.stem
    # Shorten common patterns, keep it filesystem-safe
    tag = re.sub(r"[^\w\s-]", "", stem).strip().replace(" ", "_")
    # Truncate to something reasonable
    if len(tag) > 30:
        tag = tag[:30].rstrip("_")
    return tag.lower()


def extract_clip(audio_path: Path, output_path: Path, start: float, duration: float) -> bool:
    """Extract an audio clip using ffmpeg, converting to 16kHz mono WAV."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", str(audio_path),
        "-ss", str(start),
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        "-y", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker audio clips from diarized VTT transcripts."
    )
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--vtt", help="Path to VTT file (default: same name as audio with .vtt)")
    parser.add_argument(
        "--outdir",
        default="/mnt/gdrive/Recordings/speakers",
        help="Output directory for speaker clips (default: /mnt/gdrive/Recordings/speakers)",
    )
    parser.add_argument(
        "--duration", type=float, default=15.0,
        help="Target clip duration per speaker per meeting in seconds (default: 15)",
    )
    parser.add_argument(
        "--clips-per-meeting", type=int, default=2,
        help="Max clips to extract per speaker per meeting (default: 2)",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Skip name prompts, use speaker IDs as-is",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    vtt_path = Path(args.vtt) if args.vtt else audio_path.with_suffix(".vtt")
    if not vtt_path.exists():
        print(f"Error: VTT file not found: {vtt_path}", file=sys.stderr)
        print("Use --vtt to specify the transcript file.", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)

    # Parse VTT
    print(f"Parsing: {vtt_path.name}")
    segments = parse_vtt(vtt_path)
    speakers = summarize_speakers(segments)

    if not speakers:
        print("\nNo speaker labels found in this VTT.")
        print("This transcript may not have diarization enabled.")
        print("The Whisper API URL needs diarize=true to produce speaker labels.")
        sys.exit(1)

    # Show existing enrollment profiles
    existing_profiles = []
    if outdir.exists():
        for d in sorted(outdir.iterdir()):
            if d.is_dir():
                clips = get_existing_clips(d)
                if clips:
                    total_size = sum(c.stat().st_size for c in clips) / 1024
                    existing_profiles.append(d.name)
                    print(f"  Enrolled: {d.name} ({len(clips)} clips, {total_size:.0f} KB)")

    if existing_profiles:
        print()

    # Show summary
    print(f"Found {len(speakers)} speaker(s) in {len(segments)} segments:\n")
    for spk, info in sorted(speakers.items(), key=lambda x: x[1]["total_time"], reverse=True):
        print(f"  {spk}  ({format_timestamp(info['total_time'])} total, {len(info['segments'])} segments)")
        for sample in info["sample_texts"]:
            print(f"    {sample}")
        print()

    # Map speaker IDs to real names
    name_map = {}
    if not args.non_interactive:
        if existing_profiles:
            print(f"Existing profiles: {', '.join(existing_profiles)}")
            print("(Type an existing name to append clips, or a new name to create a profile)\n")
        print("Assign real names to speakers (press Enter to skip, 'q' to quit):\n")
        for spk in sorted(speakers.keys()):
            while True:
                name = input(f"  {spk} -> Real name: ").strip()
                if name.lower() == "q":
                    print("Cancelled.")
                    sys.exit(0)
                if not name:
                    print(f"    Skipping {spk}")
                    break
                if name in name_map.values():
                    print(f"    Name '{name}' already used. Use a different name or press Enter to skip.")
                    continue
                name_map[spk] = name
                if name in existing_profiles:
                    clips = get_existing_clips(outdir / name)
                    print(f"    Will append to {name}/ (currently {len(clips)} clips)")
                else:
                    print(f"    New profile: {name}")
                break
        print()
    else:
        # Non-interactive: use all speakers with their IDs as names
        name_map = {spk: spk for spk in speakers}

    if not name_map:
        print("No speakers selected. Nothing to extract.")
        sys.exit(0)

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Extract clips for each named speaker (always uses subdirectories for append)
    source_tag = make_source_tag(audio_path)

    for spk, name in name_map.items():
        info = speakers[spk]
        best = pick_best_segments(
            info["segments"],
            target_seconds=args.duration,
            max_clips=args.clips_per_meeting,
        )

        if not best:
            print(f"  {name}: No suitable segments found (need >= 1.5s clips)")
            continue

        total_duration = sum(s["end"] - s["start"] for s in best)

        speaker_dir = outdir / name
        speaker_dir.mkdir(parents=True, exist_ok=True)
        existing = get_existing_clips(speaker_dir)
        clip_num = next_clip_number(speaker_dir)

        print(f"  {name}: extracting {len(best)} clip(s) ({format_timestamp(total_duration)} total)"
              + (f" [appending to {len(existing)} existing]" if existing else " [new profile]"))

        for seg in best:
            # e.g. clip3_feb_12_at_12-56_pm.wav
            clip_path = speaker_dir / f"clip{clip_num}_{source_tag}.wav"
            duration = seg["end"] - seg["start"]
            if extract_clip(audio_path, clip_path, seg["start"], duration):
                size_kb = clip_path.stat().st_size / 1024
                print(f"    -> {clip_path.name} ({size_kb:.0f} KB, {duration:.1f}s)")
                clip_num += 1

    # Summary
    print(f"\nDone! Clips saved to: {outdir}")
    if outdir.exists():
        for d in sorted(outdir.iterdir()):
            if d.is_dir():
                clips = get_existing_clips(d)
                if clips:
                    total_size = sum(c.stat().st_size for c in clips) / 1024
                    print(f"  {d.name}: {len(clips)} clips ({total_size:.0f} KB)")
    print("\nTo re-transcribe with speaker labels: delete the .vtt and let the scanner pick it up.")


if __name__ == "__main__":
    main()
