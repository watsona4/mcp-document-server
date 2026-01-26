#!/usr/bin/env python3
"""
MCP Document Server
Provides secure access to documents from a local directory for Claude AI
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, TextContent, Tool

# Configure logging to stderr (never stdout for STDIO transport)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s',
    handlers=[logging.StreamHandler()]  # Will use stderr by default
)
logger = logging.getLogger(__name__)
logger.info(f"Log level set to: {LOG_LEVEL}")

# Configuration from environment variables
DOCUMENTS_PATH = Path(os.getenv('DOCUMENTS_PATH', '/documents'))
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE_MB', '10')) * 1024 * 1024  # Default 10MB
ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', '.txt,.md,.pdf,.docx,.xlsx,.pptx,.csv,.json,.yaml,.yml,.log').split(',')
AUTH_TOKEN = os.getenv('MCP_AUTH_TOKEN', '')

# Initialize FastMCP server with transport security settings
from mcp.server.transport_security import TransportSecuritySettings

# Allow connections from Tailscale Funnel and localhost
ALLOWED_HOSTS = os.getenv('MCP_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

transport_security = TransportSecuritySettings(
    allowed_hosts=ALLOWED_HOSTS,
    allowed_origins=['*'],  # Allow any origin for SSE connections
)

mcp = FastMCP("Document Server", transport_security=transport_security)

logger.info(f"Document server starting with path: {DOCUMENTS_PATH}")
logger.info(f"Allowed extensions: {ALLOWED_EXTENSIONS}")


def is_safe_path(base_path: Path, requested_path: Path) -> bool:
    """Ensure requested path is within the base documents directory"""
    try:
        resolved_base = base_path.resolve()
        resolved_requested = requested_path.resolve()
        return str(resolved_requested).startswith(str(resolved_base))
    except Exception:
        return False


def get_file_info(file_path: Path) -> dict[str, Any]:
    """Get file metadata"""
    stat = file_path.stat()
    return {
        'name': file_path.name,
        'path': str(file_path.relative_to(DOCUMENTS_PATH)),
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'extension': file_path.suffix
    }


@mcp.tool()
async def list_documents(
    subdirectory: str = "",
    recursive: bool = False
) -> str:
    """
    List available documents in the document directory.

    Args:
        subdirectory: Optional subdirectory to list (relative path)
        recursive: If True, list files recursively

    Returns:
        JSON string with list of documents and their metadata
    """
    start_time = time.time()
    logger.info(f"list_documents called: subdirectory='{subdirectory}', recursive={recursive}")

    try:
        search_path = DOCUMENTS_PATH / subdirectory if subdirectory else DOCUMENTS_PATH
        logger.debug(f"Search path resolved to: {search_path}")

        if not is_safe_path(DOCUMENTS_PATH, search_path):
            logger.warning(f"Path safety check failed for: {search_path}")
            return f"Error: Access denied - path outside allowed directory"

        if not search_path.exists():
            logger.warning(f"Directory not found: {search_path}")
            return f"Error: Directory not found: {subdirectory}"

        documents = []

        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'

        logger.debug(f"Scanning with pattern: {pattern}")
        for file_path in search_path.glob(pattern):
            if file_path.is_file() and file_path.suffix in ALLOWED_EXTENSIONS:
                documents.append(get_file_info(file_path))
                logger.debug(f"Found document: {file_path.name}")

        # Sort by modification time (newest first)
        documents.sort(key=lambda x: x['modified'], reverse=True)

        elapsed = time.time() - start_time
        logger.info(f"list_documents completed: found {len(documents)} documents in {elapsed:.2f}s")

        import json
        return json.dumps({
            'directory': str(subdirectory or '/'),
            'total_files': len(documents),
            'documents': documents
        }, indent=2)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error listing documents after {elapsed:.2f}s: {e}", exc_info=True)
        return f"Error listing documents: {str(e)}"


@mcp.tool()
async def read_document(file_path: str, max_chars: int = 500000) -> str:
    """
    Read the contents of a document.

    Args:
        file_path: Relative path to the document
        max_chars: Maximum characters to return (default 500000)

    Returns:
        Document contents as string
    """
    start_time = time.time()
    logger.info(f"read_document called: file_path='{file_path}', max_chars={max_chars}")

    try:
        full_path = DOCUMENTS_PATH / file_path
        logger.debug(f"Full path resolved to: {full_path}")

        if not is_safe_path(DOCUMENTS_PATH, full_path):
            logger.warning(f"Path safety check failed for: {full_path}")
            return f"Error: Access denied - path outside allowed directory"

        if not full_path.exists():
            logger.warning(f"File not found: {full_path}")
            return f"Error: File not found: {file_path}"

        if full_path.suffix not in ALLOWED_EXTENSIONS:
            logger.warning(f"Extension not allowed: {full_path.suffix}")
            return f"Error: File type not allowed: {full_path.suffix}"

        # Check file size
        file_size = full_path.stat().st_size
        logger.debug(f"File size: {file_size} bytes")
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size} > {MAX_FILE_SIZE}")
            return f"Error: File too large ({file_size} bytes, max {MAX_FILE_SIZE})"

        # Read file based on extension
        if full_path.suffix == '.pdf':
            logger.debug("Reading PDF file")
            try:
                import pypdf
                reader = pypdf.PdfReader(full_path)
                logger.debug(f"PDF has {len(reader.pages)} pages")
                text = ""
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    logger.debug(f"Page {i+1}: extracted {len(page_text)} chars")
                content = text[:max_chars]
            except ImportError as e:
                logger.error(f"pypdf import failed: {e}")
                return "Error: PDF support not installed. Install pypdf package."
            except Exception as e:
                logger.error(f"PDF parsing error: {e}", exc_info=True)
                return f"Error reading PDF: {str(e)}"

        elif full_path.suffix in ['.docx']:
            logger.debug("Reading DOCX file")
            try:
                import docx
                doc = docx.Document(full_path)
                logger.debug(f"DOCX has {len(doc.paragraphs)} paragraphs")
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                content = text[:max_chars]
            except ImportError as e:
                logger.error(f"python-docx import failed: {e}")
                return "Error: DOCX support not installed. Install python-docx package."
            except Exception as e:
                logger.error(f"DOCX parsing error: {e}", exc_info=True)
                return f"Error reading DOCX: {str(e)}"

        elif full_path.suffix in ['.xlsx']:
            logger.debug("Reading XLSX file")
            try:
                import openpyxl
                wb = openpyxl.load_workbook(full_path)
                logger.debug(f"XLSX has {len(wb.sheetnames)} sheets: {wb.sheetnames}")
                text = f"Excel file with {len(wb.sheetnames)} sheets: {', '.join(wb.sheetnames)}\n\n"
                for sheet_name in wb.sheetnames[:5]:  # First 5 sheets
                    ws = wb[sheet_name]
                    row_count = 0
                    text += f"\n=== Sheet: {sheet_name} ===\n"
                    for row in ws.iter_rows():  # All rows (character limit still applies)
                        text += "\t".join([str(cell.value) if cell.value is not None else "" for cell in row]) + "\n"
                        row_count += 1
                    logger.debug(f"Sheet '{sheet_name}': read {row_count} rows")
                content = text[:max_chars]
            except ImportError as e:
                logger.error(f"openpyxl import failed: {e}")
                return "Error: XLSX support not installed. Install openpyxl package."
            except Exception as e:
                logger.error(f"XLSX parsing error: {e}", exc_info=True)
                return f"Error reading XLSX: {str(e)}"
        else:
            # Text-based files
            logger.debug(f"Reading text file with extension: {full_path.suffix}")
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)

        truncated = len(content) >= max_chars
        elapsed = time.time() - start_time
        logger.info(f"read_document completed: {len(content)} chars, truncated={truncated}, took {elapsed:.2f}s")

        result = {
            'file_path': file_path,
            'size': file_size,
            'extension': full_path.suffix,
            'content': content,
            'truncated': truncated
        }

        import json
        return json.dumps(result, indent=2)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error reading document {file_path} after {elapsed:.2f}s: {e}", exc_info=True)
        return f"Error reading document: {str(e)}"


@mcp.tool()
async def search_documents(
    query: str,
    file_extension: str = "",
    case_sensitive: bool = False
) -> str:
    """
    Search for documents containing specific text.

    Args:
        query: Text to search for
        file_extension: Optional file extension filter (e.g., '.txt')
        case_sensitive: Whether search should be case-sensitive

    Returns:
        JSON string with matching documents and snippets
    """
    start_time = time.time()
    logger.info(f"search_documents called: query='{query[:50]}...', extension='{file_extension}', case_sensitive={case_sensitive}")

    try:
        if not query:
            logger.warning("Empty search query")
            return "Error: Search query cannot be empty"

        search_query = query if case_sensitive else query.lower()
        matches = []
        files_scanned = 0
        files_skipped = 0

        for file_path in DOCUMENTS_PATH.rglob('*'):
            if not file_path.is_file():
                continue

            if file_path.suffix not in ALLOWED_EXTENSIONS:
                files_skipped += 1
                continue

            if file_extension and file_path.suffix != file_extension:
                files_skipped += 1
                continue

            try:
                # Only search text-based files for now
                if file_path.suffix in ['.txt', '.md', '.json', '.yaml', '.yml', '.log', '.csv']:
                    files_scanned += 1
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        search_content = content if case_sensitive else content.lower()

                        if search_query in search_content:
                            # Find snippet around match
                            index = search_content.find(search_query)
                            start = max(0, index - 100)
                            end = min(len(content), index + len(query) + 100)
                            snippet = content[start:end]

                            matches.append({
                                'path': str(file_path.relative_to(DOCUMENTS_PATH)),
                                'size': file_path.stat().st_size,
                                'snippet': snippet.strip()
                            })
                            logger.debug(f"Match found in: {file_path.name}")
                else:
                    files_skipped += 1
            except Exception as e:
                logger.warning(f"Error searching {file_path}: {e}")
                continue

        elapsed = time.time() - start_time
        logger.info(f"search_documents completed: {len(matches)} matches in {files_scanned} files ({files_skipped} skipped), took {elapsed:.2f}s")

        import json
        return json.dumps({
            'query': query,
            'total_matches': len(matches),
            'matches': matches[:50]  # Limit to 50 results
        }, indent=2)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error searching documents after {elapsed:.2f}s: {e}", exc_info=True)
        return f"Error searching documents: {str(e)}"


# Extensions that can be created/written
WRITABLE_EXTENSIONS = ['.txt', '.md', '.json', '.yaml', '.yml', '.csv', '.log']


@mcp.tool()
async def create_document(file_path: str, content: str, overwrite: bool = False) -> str:
    """
    Create a new document or update an existing one.

    Args:
        file_path: Relative path for the document (e.g., 'notes/meeting.txt')
        content: Text content to write to the file
        overwrite: If True, overwrite existing files. If False, fail if file exists.

    Returns:
        JSON string with result status and file info
    """
    start_time = time.time()
    content_size = len(content.encode('utf-8'))
    logger.info(f"create_document called: file_path='{file_path}', content_size={content_size} bytes, overwrite={overwrite}")

    try:
        import json

        # Validate file path
        if not file_path:
            logger.warning("Empty file path provided")
            return json.dumps({'success': False, 'error': 'File path cannot be empty'})

        full_path = DOCUMENTS_PATH / file_path
        logger.debug(f"Full path resolved to: {full_path}")

        # Security: ensure path is within documents directory
        if not is_safe_path(DOCUMENTS_PATH, full_path):
            logger.warning(f"Path safety check failed for: {full_path}")
            return json.dumps({'success': False, 'error': 'Invalid path - must be within documents directory'})

        # Check extension is writable
        if full_path.suffix not in WRITABLE_EXTENSIONS:
            logger.warning(f"Extension not writable: {full_path.suffix}")
            return json.dumps({
                'success': False,
                'error': f'Extension {full_path.suffix} not allowed. Allowed: {", ".join(WRITABLE_EXTENSIONS)}'
            })

        # Check if file exists
        file_existed = full_path.exists()
        if file_existed and not overwrite:
            logger.warning(f"File exists and overwrite=False: {full_path}")
            return json.dumps({
                'success': False,
                'error': f'File already exists: {file_path}. Set overwrite=True to replace it.'
            })

        # Create parent directories if needed
        if not full_path.parent.exists():
            logger.debug(f"Creating parent directories: {full_path.parent}")
            full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        logger.debug(f"Writing {content_size} bytes to {full_path}")
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        elapsed = time.time() - start_time
        logger.info(f"create_document completed: wrote {content_size} bytes to {file_path}, overwritten={file_existed and overwrite}, took {elapsed:.2f}s")

        return json.dumps({
            'success': True,
            'file_path': file_path,
            'size': content_size,
            'overwritten': file_existed and overwrite
        }, indent=2)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error creating document {file_path} after {elapsed:.2f}s: {e}", exc_info=True)
        return json.dumps({'success': False, 'error': str(e)})


@mcp.resource("documents://list")
def list_documents_resource() -> str:
    """Resource that provides a list of all available documents"""
    documents = []
    for file_path in DOCUMENTS_PATH.rglob('*'):
        if file_path.is_file() and file_path.suffix in ALLOWED_EXTENSIONS:
            documents.append(get_file_info(file_path))
    
    import json
    return json.dumps({
        'total': len(documents),
        'documents': documents[:100]  # Limit to 100 for resource
    }, indent=2)


def main():
    """Main entry point"""
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response

    # Ensure documents directory exists
    DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)

    # Log startup configuration
    logger.info("=" * 60)
    logger.info("MCP Document Server Starting")
    logger.info("=" * 60)
    logger.info(f"Documents path: {DOCUMENTS_PATH}")
    logger.info(f"Documents path exists: {DOCUMENTS_PATH.exists()}")
    logger.info(f"Documents path writable: {os.access(DOCUMENTS_PATH, os.W_OK)}")
    logger.info(f"Max file size: {MAX_FILE_SIZE / 1024 / 1024:.1f} MB")
    logger.info(f"Allowed extensions: {ALLOWED_EXTENSIONS}")
    logger.info(f"Writable extensions: {WRITABLE_EXTENSIONS}")
    logger.info(f"Allowed hosts: {ALLOWED_HOSTS}")
    logger.info(f"Auth token configured: {bool(AUTH_TOKEN)}")
    logger.info(f"Log level: {LOG_LEVEL}")

    # Count existing documents
    try:
        doc_count = sum(1 for f in DOCUMENTS_PATH.rglob('*') if f.is_file() and f.suffix in ALLOWED_EXTENSIONS)
        logger.info(f"Found {doc_count} existing documents")
    except Exception as e:
        logger.warning(f"Could not count documents: {e}")

    # Check which transport to use based on environment
    transport = os.getenv('MCP_TRANSPORT', 'sse')

    if transport == 'stdio':
        logger.info("Using STDIO transport")
        mcp.run(transport='stdio')
    else:
        # Default to SSE for HTTP-based access
        host = os.getenv('MCP_HOST', '0.0.0.0')
        port = int(os.getenv('MCP_PORT', '8000'))
        logger.info(f"Using SSE transport on {host}:{port}")

        app = mcp.sse_app()

        # Add health endpoint
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route

        async def health(request):
            return JSONResponse({'status': 'healthy'})

        app.routes.append(Route('/health', health))

        # Add authentication middleware if token is configured
        if AUTH_TOKEN:
            logger.info(f"Authentication enabled (token length: {len(AUTH_TOKEN)})")

            # Use pure ASGI middleware (BaseHTTPMiddleware breaks SSE streaming)
            from starlette.datastructures import URL, QueryParams

            class AuthMiddleware:
                def __init__(self, app):
                    self.app = app

                async def __call__(self, scope, receive, send):
                    if scope["type"] != "http":
                        return await self.app(scope, receive, send)

                    path = scope.get("path", "")
                    method = scope.get("method", "UNKNOWN")
                    client = scope.get("client", ("unknown", 0))
                    client_ip = client[0] if client else "unknown"

                    logger.debug(f"Request: {method} {path} from {client_ip}")

                    # Allow healthcheck and messages endpoints without token
                    # (messages require valid session_id from authenticated SSE)
                    if path == "/health":
                        logger.debug(f"Health check from {client_ip}")
                        return await self.app(scope, receive, send)

                    if path.startswith("/messages"):
                        logger.debug(f"Messages endpoint from {client_ip} (session auth)")
                        return await self.app(scope, receive, send)

                    # Check Authorization header
                    headers = dict(scope.get("headers", []))
                    auth_header = headers.get(b"authorization", b"").decode()
                    if auth_header == f"Bearer {AUTH_TOKEN}":
                        logger.debug(f"Authenticated via Bearer token from {client_ip}")
                        return await self.app(scope, receive, send)

                    # Check token query parameter
                    query_string = scope.get("query_string", b"").decode()
                    query_params = QueryParams(query_string)
                    if query_params.get("token") == AUTH_TOKEN:
                        logger.debug(f"Authenticated via query param from {client_ip}")
                        return await self.app(scope, receive, send)

                    # Unauthorized - log details for debugging
                    has_auth_header = bool(auth_header)
                    has_query_token = "token" in query_params
                    logger.warning(
                        f"Unauthorized request: {method} {path} from {client_ip} "
                        f"(has_auth_header={has_auth_header}, has_query_token={has_query_token})"
                    )
                    response = Response(content="Unauthorized", status_code=401)
                    await response(scope, receive, send)

            app = AuthMiddleware(app)
        else:
            logger.warning("No MCP_AUTH_TOKEN set - server is unauthenticated!")

        uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
