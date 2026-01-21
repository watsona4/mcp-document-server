# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP Document Server - A Python FastAPI-based Model Context Protocol (MCP) server that provides Claude AI with secure, read-only access to local documents. Supports both network (SSE/HTTP) and local (STDIO) transports.

## Commands

### Docker (Recommended)
```bash
docker-compose up -d              # Start server
docker-compose down               # Stop server
docker-compose logs -f            # View logs
docker-compose up -d --build      # Rebuild and start
```

### Direct Python
```bash
python -m venv venv
venv\Scripts\activate             # Windows
source venv/bin/activate          # Linux/Mac
pip install -r requirements.txt
python server.py                  # Run server
```

### Testing
```bash
curl http://localhost:8000/health # Health check
python test_server.py             # Run test client
```

## Architecture

```
Claude AI
    │
    ├── STDIO (local) ──┐
    │                   │
    └── SSE/HTTP ───────┴──→ MCP Document Server (FastMCP + FastAPI)
                                    │
                                    ├── list_documents()   - List files with metadata
                                    ├── read_document()    - Read with format parsing
                                    ├── search_documents() - Full-text search
                                    └── documents://list   - Resource endpoint
                                    │
                                    ↓
                            Documents Directory (read-only)
```

### Core Components

**server.py** - Main implementation:
- `is_safe_path()` - Path traversal protection (validates all paths stay within DOCUMENTS_PATH)
- `get_file_info()` - File metadata extraction
- `list_documents()` / `read_document()` / `search_documents()` - MCP tools
- `main()` - Entry point with transport selection (SSE vs STDIO)

**test_server.py** - Async test client that validates all MCP tools via Docker exec

### Document Parsing
- PDF: pypdf text extraction
- DOCX: python-docx paragraph extraction
- XLSX: openpyxl sheet enumeration + first 100 rows of first 5 sheets
- Text formats: Direct read (TXT, MD, CSV, JSON, YAML, LOG)

## Configuration

Environment variables (see .env.example):
- `MCP_TRANSPORT` - 'sse' or 'stdio'
- `MCP_HOST` / `MCP_PORT` - Server binding (SSE only)
- `DOCUMENTS_PATH` - Documents directory
- `MAX_FILE_SIZE_MB` - Size limit (default 10MB)
- `ALLOWED_EXTENSIONS` - Whitelist of file types

## Security Model

1. **Path Traversal Protection**: All paths resolved to absolute and validated within DOCUMENTS_PATH
2. **Read-Only Access**: Docker volume mounted with `:ro`, no write operations exposed
3. **Extension Whitelist**: Only allowed file types can be accessed
4. **Container Isolation**: Resource limits, no-new-privileges, non-root capable

## Key Design Decisions

- Logs go to stderr only (stdout reserved for STDIO protocol)
- Async throughout using FastAPI's async patterns
- Security validation happens before any file operation
- File size limits enforced at read time, not listing
