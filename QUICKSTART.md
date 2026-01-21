# MCP Document Server - Quick Start

Get your document server running in 5 minutes!

## 🚀 Quick Setup

### 1. Install (choose one method)

**Option A: Automated (Recommended)**
```bash
cd mcp-document-server
chmod +x install.sh
./install.sh
```

**Option B: Manual**
```bash
# Copy environment template
cp .env.example .env

# Create documents directory
mkdir -p documents

# Build Docker image
docker-compose build
```

### 2. Add Your Documents

```bash
# Copy work documents to the documents folder
cp /path/to/your/work/docs/*.pdf documents/
cp /path/to/your/work/docs/*.docx documents/

# Or create a symbolic link
ln -s /path/to/your/work/docs documents/work

# Verify files are there
ls -la documents/
```

### 3. Start the Server

```bash
docker-compose up -d
```

### 4. Test It

```bash
# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f

# Run test script
python test_server.py
```

### 5. Connect to Claude

Edit Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "document-server": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-document-server", "python", "server.py"],
      "env": {"MCP_TRANSPORT": "stdio"}
    }
  }
}
```

Restart Claude Desktop and ask:
```
Can you list my documents?
```

## 🎯 Common Workflows

### Work Documents on Home Server + Tailscale

Perfect for Aaron's use case!

1. **On your home server**:
   ```bash
   # Install in a permanent location
   sudo mkdir -p /opt/mcp-document-server
   sudo cp -r . /opt/mcp-document-server
   cd /opt/mcp-document-server
   
   # Mount your work documents
   # Edit docker-compose.yml volume:
   # - /home/aaron/work-docs:/documents:ro
   
   # Start server
   docker-compose up -d
   
   # Install systemd service
   sudo cp mcp-document-server.service /etc/systemd/system/
   sudo systemctl enable mcp-document-server
   sudo systemctl start mcp-document-server
   ```

2. **Get Tailscale IP**:
   ```bash
   tailscale ip -4
   # Example output: 100.x.x.x
   ```

3. **Configure OPNsense firewall** (if needed):
   - Navigate to Firewall > Rules > Tailscale
   - Add rule: Allow TCP port 8000 from Tailscale network

4. **Access from anywhere**:
   - Server available at: `http://100.x.x.x:8000`
   - Test: `curl http://100.x.x.x:8000/health`

### Local Development

1. **Run with Python**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   export DOCUMENTS_PATH=./documents
   python server.py
   ```

2. **Hot reload for development**:
   ```bash
   # Install development dependencies
   pip install watchdog
   
   # Run with auto-reload
   MCP_TRANSPORT=sse python server.py --reload
   ```

## 🔧 Useful Commands

### Docker Management
```bash
# Start server
docker-compose up -d

# Stop server
docker-compose down

# Restart server
docker-compose restart

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Update after code changes
docker-compose up -d --build
```

### Systemd Management (if installed)
```bash
# Start
sudo systemctl start mcp-document-server

# Stop
sudo systemctl stop mcp-document-server

# Restart
sudo systemctl restart mcp-document-server

# Status
sudo systemctl status mcp-document-server

# View logs
sudo journalctl -u mcp-document-server -f
```

### Testing
```bash
# Test HTTP endpoint
curl http://localhost:8000/health

# Test with full script
python test_server.py

# List documents via API
curl http://localhost:8000/mcp -X POST \
  -H "Content-Type: application/json" \
  -d '{"tool":"list_documents"}'
```

## 🐛 Quick Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
sudo lsof -i :8000

# Check Docker
docker ps -a | grep mcp

# Check logs
docker-compose logs
```

### Can't read files
```bash
# Check documents exist
ls -la documents/

# Check permissions
chmod 755 documents/
chmod 644 documents/*

# Verify extensions allowed
cat .env | grep ALLOWED
```

### Claude can't connect
```bash
# Check server is running
docker ps | grep mcp

# Check config syntax
cat ~/.config/claude/claude_desktop_config.json | jq .

# Test manually
python test_server.py
```

## 📚 Next Steps

- Read [README.md](README.md) for complete documentation
- See [CLAUDE_INTEGRATION.md](CLAUDE_INTEGRATION.md) for Claude setup
- Check logs if issues: `docker-compose logs -f`

## 💡 Tips

1. **Use read-only mounts** for security:
   ```yaml
   volumes:
     - ./documents:/documents:ro  # :ro = read-only
   ```

2. **Organize documents** by project:
   ```
   documents/
   ├── projects/
   │   ├── 2024/
   │   └── 2025/
   ├── reports/
   └── planning/
   ```

3. **Set appropriate file limits** in `.env`:
   ```bash
   MAX_FILE_SIZE_MB=20  # Increase for larger files
   ```

4. **Monitor logs** occasionally:
   ```bash
   docker-compose logs --tail=100
   ```

---

**Ready to use!** 🎉

Ask Claude: "Can you list my work documents?"
