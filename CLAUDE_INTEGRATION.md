# Claude Desktop Integration Guide

This guide shows you how to connect your MCP Document Server to Claude Desktop so Claude can access your work documents.

## Prerequisites

- MCP Document Server running (via Docker or Python)
- Claude Desktop application installed
- Documents directory populated with files

## Method 1: Local STDIO Connection

This method is recommended for local development.

### Step 1: Locate Claude Desktop Config

The configuration file location depends on your OS:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Step 2: Edit Configuration

Create or edit the config file:

```json
{
  "mcpServers": {
    "document-server": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "mcp-document-server",
        "python",
        "server.py"
      ],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "DOCUMENTS_PATH": "/documents"
      }
    }
  }
}
```

### Step 3: Restart Claude Desktop

1. Quit Claude Desktop completely
2. Restart the application
3. Look for the MCP server indicator

### Step 4: Test the Connection

In Claude Desktop, try:

```
Can you list the documents available in my document server?
```

Claude should now be able to access your MCP server!

## Method 2: Remote HTTP/SSE Connection

This method is better for remote access (e.g., via Tailscale).

### Step 1: Get Your Server URL

If using Tailscale:

```bash
# Get your Tailscale IP
tailscale ip -4

# Your server URL will be:
# http://<tailscale-ip>:8000
```

### Step 2: Configure Claude Desktop

Edit the Claude Desktop config:

```json
{
  "mcpServers": {
    "document-server": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-everything",
        "sse",
        "http://<your-tailscale-ip>:8000/sse"
      ]
    }
  }
}
```

Replace `<your-tailscale-ip>` with your actual IP.

### Step 3: Restart and Test

Same as Method 1, Step 3-4.

## Method 3: Custom MCP Client Script

For more control, create a custom client script.

### Step 1: Create Client Script

Save as `~/mcp-document-client.sh`:

```bash
#!/bin/bash
docker exec -i mcp-document-server python server.py
```

Make it executable:

```bash
chmod +x ~/mcp-document-client.sh
```

### Step 2: Configure Claude Desktop

```json
{
  "mcpServers": {
    "document-server": {
      "command": "/home/yourusername/mcp-document-client.sh"
    }
  }
}
```

## Troubleshooting

### Server Not Appearing in Claude

1. **Check config syntax**: Validate JSON syntax
   ```bash
   # On Linux/Mac
   cat ~/.config/claude/claude_desktop_config.json | jq .
   ```

2. **Check server is running**:
   ```bash
   docker ps | grep mcp-document-server
   ```

3. **Check logs**:
   ```bash
   docker-compose logs -f
   ```

### Claude Can't Access Documents

1. **Verify documents exist**:
   ```bash
   ls -la ./documents/
   ```

2. **Check permissions**:
   ```bash
   # Documents should be readable
   chmod 644 ./documents/*
   ```

3. **Test manually**:
   ```bash
   python test_server.py
   ```

### Connection Timeout

1. **For STDIO**: Ensure Docker container is running
2. **For HTTP**: Check firewall rules
3. **For Tailscale**: Verify connectivity
   ```bash
   ping <tailscale-ip>
   curl http://<tailscale-ip>:8000/health
   ```

## Advanced Configuration

### Multiple Document Servers

You can run multiple MCP servers for different document sets:

```json
{
  "mcpServers": {
    "work-documents": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-work-docs", "python", "server.py"],
      "env": {"DOCUMENTS_PATH": "/work-documents"}
    },
    "personal-documents": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-personal-docs", "python", "server.py"],
      "env": {"DOCUMENTS_PATH": "/personal-documents"}
    }
  }
}
```

### With Authentication

If you add authentication to your server:

```json
{
  "mcpServers": {
    "document-server": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-document-server", "python", "server.py"],
      "env": {
        "MCP_AUTH_TOKEN": "your-secret-token-here"
      }
    }
  }
}
```

## Usage Examples

Once configured, you can ask Claude:

### List Documents
```
Can you list all the documents in my work folder?
```

### Read a Document
```
Can you read the contents of projects/Q4-report.pdf?
```

### Search Documents
```
Search my documents for "quarterly review" 
```

### Analyze Documents
```
Read the strategic plan document and summarize the key initiatives for FY26
```

## Security Notes

- Documents are **read-only** - Claude cannot modify them
- Path traversal is prevented - Claude cannot access outside DOCUMENTS_PATH
- Consider using Tailscale for encrypted remote access
- Use authentication tokens for production deployments

## Additional Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude Desktop Documentation](https://claude.ai/desktop)
- [Tailscale Setup Guide](https://tailscale.com/kb/)

---

**Questions?** Check the main README.md or server logs for troubleshooting.
