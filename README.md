# MCP Document Server

A secure Model Context Protocol (MCP) server that provides Claude AI with access to your local documents. Built for Aaron's work document integration needs.

## Features

- 🔒 **Secure**: Read-only access with path traversal protection
- 📁 **File Support**: PDF, DOCX, XLSX, PPTX, TXT, MD, JSON, YAML, CSV, LOG
- 🔍 **Search**: Full-text search across documents
- 🐳 **Docker**: Easy deployment with Docker Compose
- 🌐 **Network**: Works over HTTP (SSE) or local STDIO
- 🔌 **Tailscale Ready**: Integrate with your existing Tailscale network

## Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone and setup**:
   ```bash
   cd mcp-document-server
   cp .env.example .env
   # Edit .env if needed
   ```

2. **Create documents directory**:
   ```bash
   mkdir -p documents
   # Add your work documents here
   ```

3. **Start the server**:
   ```bash
   docker-compose up -d
   ```

4. **Test it**:
   ```bash
   curl http://localhost:8000/health
   ```

### Option 2: Python Virtual Environment

1. **Setup**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure**:
   ```bash
   cp .env.example .env
   export $(cat .env | xargs)
   ```

3. **Run**:
   ```bash
   python server.py
   ```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `sse` | Transport type: `sse` (HTTP) or `stdio` |
| `MCP_HOST` | `0.0.0.0` | Server host (SSE only) |
| `MCP_PORT` | `8000` | Server port (SSE only) |
| `DOCUMENTS_PATH` | `/documents` | Path to documents directory |
| `MAX_FILE_SIZE_MB` | `10` | Maximum file size in MB |
| `ALLOWED_EXTENSIONS` | `.txt,.md,...` | Comma-separated allowed extensions |

### Docker Compose Volumes

Edit `docker-compose.yml` to mount your work documents:

```yaml
volumes:
  # Option 1: Local directory
  - ./documents:/documents:ro
  
  # Option 2: Specific path
  - /path/to/work/documents:/documents:ro
```

## Integration with Claude

### Via Claude Desktop (Local)

1. Edit Claude Desktop config (`~/.config/claude/mcp_servers.json` on Linux):

```json
{
  "document-server": {
    "command": "docker",
    "args": ["exec", "-i", "mcp-document-server", "python", "server.py"],
    "env": {
      "MCP_TRANSPORT": "stdio"
    }
  }
}
```

### Via HTTP (Tailscale/Network)

If running on your home server and accessing via Tailscale:

1. Start server with SSE transport (default)
2. Note your Tailscale IP (e.g., `100.x.x.x`)
3. Access at `http://100.x.x.x:8000`

Configure Claude to use this URL in your MCP client settings.

## Available Tools

### 1. `list_documents`

List all documents in a directory.

**Parameters**:
- `subdirectory` (optional): Subdirectory to list
- `recursive` (optional): List recursively

**Example**:
```json
{
  "subdirectory": "projects/2024",
  "recursive": true
}
```

### 2. `read_document`

Read the contents of a specific document.

**Parameters**:
- `file_path`: Relative path to document
- `max_chars` (optional): Maximum characters to return (default: 50000)

**Example**:
```json
{
  "file_path": "projects/2024/Q4-report.pdf",
  "max_chars": 50000
}
```

### 3. `search_documents`

Search for documents containing specific text.

**Parameters**:
- `query`: Search query
- `file_extension` (optional): Filter by extension
- `case_sensitive` (optional): Case-sensitive search

**Example**:
```json
{
  "query": "quarterly review",
  "file_extension": ".docx",
  "case_sensitive": false
}
```

## Security Considerations

### Path Traversal Protection
- All file access is restricted to `DOCUMENTS_PATH`
- Symbolic links are resolved and validated
- Path traversal attempts (../) are blocked

### Read-Only Access
- Server only reads files, never writes
- Docker container runs with read-only volume mount
- No file modification capabilities exposed

### File Type Restrictions
- Only allowed extensions can be accessed
- Binary files require appropriate parsers
- Unknown types are rejected

### Network Security
- Use Tailscale for encrypted access
- Consider adding authentication token
- Bind to localhost if only local access needed

## Tailscale Integration

### Setup

1. **On your home server**, ensure Tailscale is running:
   ```bash
   sudo tailscale status
   ```

2. **Start MCP server**:
   ```bash
   docker-compose up -d
   ```

3. **Get Tailscale IP**:
   ```bash
   tailscale ip -4
   ```

4. **Access from anywhere**:
   ```
   http://<your-tailscale-ip>:8000
   ```

### Firewall Rules

If using OPNsense with Tailscale:

1. Allow port 8000 on Tailscale interface
2. Add rule: `pass in on tailscale0 proto tcp from any to any port 8000`

## Advanced Configuration

### Custom Document Parsers

To add support for additional file types, modify `server.py`:

```python
elif full_path.suffix == '.custom':
    # Your custom parser here
    content = parse_custom_file(full_path)
```

### Authentication

Add basic token authentication:

1. Set `MCP_AUTH_TOKEN` in `.env`
2. Modify server to check token in requests
3. Pass token in Claude MCP client config

### Multiple Document Directories

Run multiple instances with different compose files:

```bash
# Work documents
docker-compose -f docker-compose.work.yml up -d

# Personal documents
docker-compose -f docker-compose.personal.yml up -d
```

## Troubleshooting

### Server won't start

```bash
# Check logs
docker-compose logs -f

# Verify documents directory exists
ls -la ./documents

# Check permissions
chmod 755 ./documents
```

### Cannot read files

```bash
# Verify file extensions are allowed
docker-compose exec mcp-document-server python -c \
  "import os; print(os.getenv('ALLOWED_EXTENSIONS'))"

# Check file permissions
ls -la ./documents/
```

### Claude can't connect

```bash
# Test server is running
curl http://localhost:8000/health

# Check Tailscale connectivity
ping <tailscale-ip>

# Verify firewall rules
# On OPNsense, check Firewall > Rules > Tailscale
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python server.py
```

### Hot Reload

For development with auto-reload:

```bash
uvicorn server:mcp.app --reload --host 0.0.0.0 --port 8000
```

## Architecture

```
┌─────────────┐
│   Claude    │
│  (claude.ai)│
└──────┬──────┘
       │ MCP Protocol
       │ (SSE/HTTP)
       ↓
┌─────────────────┐      ┌──────────────┐
│  MCP Document   │─────→│  Documents   │
│     Server      │      │  Directory   │
│   (Docker)      │      │ (Read-Only)  │
└─────────────────┘      └──────────────┘
       │
       │ Tailscale Network
       │
┌─────────────────┐
│  Home Server    │
│  (OPNsense)     │
└─────────────────┘
```

## Contributing

This is a custom implementation for Aaron's needs, but feel free to adapt it for your own use.

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- Check troubleshooting section
- Review Docker logs: `docker-compose logs -f`
- Verify network connectivity with Tailscale

---

**Built for**: Aaron Watson  
**Integration**: Claude AI + Local Documents  
**Stack**: Python, MCP SDK, Docker, FastAPI
