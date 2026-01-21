# OPNsense + Tailscale Integration Guide

Detailed guide for integrating MCP Document Server with your OPNsense firewall and Tailscale setup.

## Architecture Overview

```
┌──────────────┐
│ Work Laptop  │
│ (Tailscale)  │
└──────┬───────┘
       │
       │ Tailscale VPN
       │ (Encrypted)
       │
┌──────▼────────────────────────┐
│      Home Network             │
│  ┌─────────────────────────┐  │
│  │   OPNsense Firewall     │  │
│  │   - Kea DHCP            │  │
│  │   - Tailscale Plugin    │  │
│  │   - Firewall Rules      │  │
│  └───────────┬─────────────┘  │
│              │                 │
│  ┌───────────▼─────────────┐  │
│  │   Docker Host           │  │
│  │   - MCP Server (8000)   │  │
│  │   - Home Assistant      │  │
│  │   - Other Services      │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
```

## Prerequisites

- OPNsense firewall with Tailscale plugin installed
- Docker host on your home network
- Tailscale account and authentication

## Step 1: Tailscale Setup on OPNsense

### Install Tailscale Plugin

1. **Install via GUI**:
   - Navigate to: System > Firmware > Plugins
   - Search for: `os-tailscale`
   - Click Install

2. **Or via CLI**:
   ```bash
   pkg install os-tailscale
   ```

### Configure Tailscale

1. Navigate to: VPN > Tailscale > Settings
2. Enable Tailscale
3. Authenticate with your Tailscale account
4. Note your OPNsense Tailscale IP (e.g., 100.x.x.x)

### Enable Exit Node (Optional)

If you want to route all traffic through your home network:

1. VPN > Tailscale > Settings
2. Check "Exit Node"
3. Check "Advertise Routes"
4. Add your LAN subnet (e.g., 192.168.1.0/24)

## Step 2: MCP Server Deployment

### On Your Docker Host

1. **Install MCP Server**:
   ```bash
   cd /opt
   sudo git clone <your-repo> mcp-document-server
   cd mcp-document-server
   ```

2. **Configure Environment**:
   ```bash
   sudo cp .env.example .env
   sudo nano .env
   ```
   
   Edit `.env`:
   ```bash
   MCP_TRANSPORT=sse
   MCP_HOST=0.0.0.0
   MCP_PORT=8000
   DOCUMENTS_PATH=/opt/work-documents
   ```

3. **Setup Documents Directory**:
   ```bash
   # Option 1: Local directory
   sudo mkdir -p /opt/work-documents
   sudo chmod 755 /opt/work-documents
   
   # Option 2: Mount network share
   sudo mkdir -p /mnt/nas/work-docs
   sudo mount -t cifs //nas.local/work /mnt/nas/work-docs
   
   # Update docker-compose.yml accordingly
   ```

4. **Start Server**:
   ```bash
   sudo docker-compose up -d
   ```

5. **Get Docker Host IP**:
   ```bash
   # Get LAN IP
   ip addr show | grep "inet "
   # Example: 192.168.1.50
   ```

## Step 3: OPNsense Firewall Rules

### Create Firewall Rules for MCP Server

1. **Navigate to**: Firewall > Rules > LAN (or your appropriate interface)

2. **Add Rule for Local Access**:
   - Action: Pass
   - Interface: LAN
   - Protocol: TCP
   - Source: LAN net
   - Destination: Single host (192.168.1.50)
   - Destination Port: 8000
   - Description: MCP Document Server - Local Access

3. **Add Rule for Tailscale Access**:
   - Navigate to: Firewall > Rules > Tailscale
   - Add Rule:
     - Action: Pass
     - Interface: Tailscale
     - Protocol: TCP
     - Source: Tailscale net
     - Destination: Single host (192.168.1.50)
     - Destination Port: 8000
     - Description: MCP Document Server - Tailscale Access

4. **Apply Changes**: Click "Apply Changes"

### Verify Rules

```bash
# On OPNsense
pfctl -sr | grep 8000

# Test from Tailscale network
curl http://192.168.1.50:8000/health
```

## Step 4: Kea DHCP Reservation (Optional)

To ensure your Docker host always has the same IP:

1. **Navigate to**: Services > Kea DHCPv4 > Reservations

2. **Add Reservation**:
   - MAC Address: <docker-host-mac>
   - IP Address: 192.168.1.50
   - Hostname: docker-host
   - Description: MCP Server Host

3. **Apply Configuration**

## Step 5: DNS Configuration

### Unbound Local Records

1. **Navigate to**: Services > Unbound DNS > Overrides > Host Overrides

2. **Add Override**:
   - Host: mcp
   - Domain: home.local (or your domain)
   - IP: 192.168.1.50
   - Description: MCP Document Server

3. Now accessible at: `http://mcp.home.local:8000`

### Optional: Add to AdGuard Home

If using AdGuard Home for DNS:

1. Navigate to AdGuard Home interface
2. Filters > DNS rewrites
3. Add rewrite:
   - Domain: mcp.home.local
   - IP: 192.168.1.50

## Step 6: Testing

### From OPNsense

```bash
# SSH into OPNsense
ssh root@opnsense.local

# Test connectivity
ping 192.168.1.50
curl http://192.168.1.50:8000/health

# Test via Tailscale
curl http://mcp.home.local:8000/health
```

### From Work Laptop (via Tailscale)

```bash
# Get your Tailscale status
tailscale status

# Test MCP server
curl http://192.168.1.50:8000/health

# Or via DNS
curl http://mcp.home.local:8000/health
```

### From Claude Desktop

Configure `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "work-documents": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-everything",
        "sse",
        "http://mcp.home.local:8000/sse"
      ]
    }
  }
}
```

## Advanced Configuration

### NAT Port Forward (External Access)

**⚠️ Warning**: Only do this if you need external access and understand the security implications!

1. Navigate to: Firewall > NAT > Port Forward
2. Add Rule:
   - Interface: WAN
   - Protocol: TCP
   - Destination: WAN address
   - Destination Port: 8000
   - Redirect Target IP: 192.168.1.50
   - Redirect Target Port: 8000
   - Description: MCP Server (HTTPS recommended!)

**Better approach**: Use Cloudflare Tunnel instead:

```bash
# On Docker host
docker run -d \
  --name cloudflared \
  cloudflare/cloudflared:latest \
  tunnel --no-autoupdate run \
  --token <your-tunnel-token>
```

### Monitoring with Zenarmor

If using Zenarmor (Sensei):

1. Navigate to: Services > Zenarmor > Dashboard
2. Add application policy for MCP traffic
3. Monitor connections to port 8000

### Logging

Enable detailed logging:

1. Navigate to: Firewall > Log Files > Settings
2. Enable logging for your MCP rules
3. View logs: Firewall > Log Files > Live View

Filter for MCP traffic:
```
dst port 8000
```

## Integration with Home Assistant

Since you're running Home Assistant, you can monitor MCP server:

```yaml
# configuration.yaml
sensor:
  - platform: rest
    name: MCP Server Status
    resource: http://192.168.1.50:8000/health
    value_template: '{{ value_json.status }}'
    scan_interval: 60
    
binary_sensor:
  - platform: ping
    name: MCP Server
    host: 192.168.1.50
    port: 8000
```

## Troubleshooting

### Can't Access from Tailscale

1. **Check Tailscale status on OPNsense**:
   ```bash
   tailscale status
   ```

2. **Verify firewall rules**:
   ```bash
   pfctl -sr | grep tailscale
   ```

3. **Check if subnet routes are advertised**:
   - VPN > Tailscale > Settings
   - Verify "Advertise Routes" includes your LAN subnet

### Connection Timeout

1. **Check if server is running**:
   ```bash
   docker ps | grep mcp-document-server
   ```

2. **Test locally on Docker host**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Check OPNsense firewall logs**:
   - Firewall > Log Files > Live View
   - Look for blocks on port 8000

### DNS Not Resolving

1. **Check Unbound**:
   - Services > Unbound DNS > General
   - Verify service is running

2. **Test DNS resolution**:
   ```bash
   nslookup mcp.home.local
   dig mcp.home.local
   ```

3. **Restart Unbound**:
   - Services > Unbound DNS > General
   - Click "Restart"

## Security Best Practices

1. **Use Tailscale** for remote access (encrypted by default)
2. **Avoid port forwarding** on WAN interface
3. **Keep documents read-only** in Docker mount
4. **Enable logging** for audit trail
5. **Regular updates**:
   ```bash
   cd /opt/mcp-document-server
   git pull
   docker-compose up -d --build
   ```

6. **Monitor with CrowdSec** (if installed):
   - Automatically blocks suspicious traffic
   - Integrates with OPNsense firewall

## Network Performance Tips

1. **Use VLANs** if separating traffic:
   ```
   VLAN 10: Management (OPNsense, MCP Server)
   VLAN 20: User devices
   VLAN 30: Tailscale
   ```

2. **QoS Rules** for MCP traffic (if needed):
   - Firewall > Shaper > Rules
   - Prioritize port 8000 traffic

3. **Enable Jumbo Frames** (if using 10GbE):
   - Interfaces > Settings
   - Set MTU to 9000 on appropriate interfaces

## Backup Configuration

### OPNsense Config Backup

```bash
# Manual backup
System > Configuration > Backups

# Automated backup
System > Configuration > Backups > Automated Backup
# Enable automatic backups to Nextcloud/Google Drive
```

### MCP Server Backup

```bash
#!/bin/bash
# backup-mcp.sh
tar -czf /backup/mcp-server-$(date +%Y%m%d).tar.gz \
  /opt/mcp-document-server/.env \
  /opt/mcp-document-server/docker-compose.yml

# Keep only last 7 days
find /backup -name "mcp-server-*.tar.gz" -mtime +7 -delete
```

---

**Your Setup Summary**:
- OPNsense firewall with Tailscale plugin
- Docker host running MCP server on port 8000
- Accessible via Tailscale from anywhere
- Local access via mcp.home.local
- Integrated with existing Home Assistant setup

Need help? Check firewall logs and Docker logs first!
