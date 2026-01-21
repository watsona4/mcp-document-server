#!/usr/bin/env python3
"""
MCP Document Server
Provides secure access to documents from a local directory for Claude AI
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, TextContent, Tool

# Configure logging to stderr (never stdout for STDIO transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Will use stderr by default
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
DOCUMENTS_PATH = Path(os.getenv('DOCUMENTS_PATH', '/documents'))
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE_MB', '10')) * 1024 * 1024  # Default 10MB
ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', '.txt,.md,.pdf,.docx,.xlsx,.pptx,.csv,.json,.yaml,.yml,.log').split(',')

# Initialize FastMCP server
mcp = FastMCP("Document Server")

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
    try:
        search_path = DOCUMENTS_PATH / subdirectory if subdirectory else DOCUMENTS_PATH
        
        if not is_safe_path(DOCUMENTS_PATH, search_path):
            return f"Error: Access denied - path outside allowed directory"
        
        if not search_path.exists():
            return f"Error: Directory not found: {subdirectory}"
        
        documents = []
        
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'
        
        for file_path in search_path.glob(pattern):
            if file_path.is_file() and file_path.suffix in ALLOWED_EXTENSIONS:
                documents.append(get_file_info(file_path))
        
        # Sort by modification time (newest first)
        documents.sort(key=lambda x: x['modified'], reverse=True)
        
        import json
        return json.dumps({
            'directory': str(subdirectory or '/'),
            'total_files': len(documents),
            'documents': documents
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        return f"Error listing documents: {str(e)}"


@mcp.tool()
async def read_document(file_path: str, max_chars: int = 50000) -> str:
    """
    Read the contents of a document.
    
    Args:
        file_path: Relative path to the document
        max_chars: Maximum characters to return (default 50000)
    
    Returns:
        Document contents as string
    """
    try:
        full_path = DOCUMENTS_PATH / file_path
        
        if not is_safe_path(DOCUMENTS_PATH, full_path):
            return f"Error: Access denied - path outside allowed directory"
        
        if not full_path.exists():
            return f"Error: File not found: {file_path}"
        
        if full_path.suffix not in ALLOWED_EXTENSIONS:
            return f"Error: File type not allowed: {full_path.suffix}"
        
        # Check file size
        file_size = full_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return f"Error: File too large ({file_size} bytes, max {MAX_FILE_SIZE})"
        
        # Read file based on extension
        if full_path.suffix == '.pdf':
            # For PDFs, we'll need PyPDF2 or similar
            try:
                import pypdf
                reader = pypdf.PdfReader(full_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                content = text[:max_chars]
            except ImportError:
                return "Error: PDF support not installed. Install pypdf package."
                
        elif full_path.suffix in ['.docx']:
            # For Word docs, we'll need python-docx
            try:
                import docx
                doc = docx.Document(full_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                content = text[:max_chars]
            except ImportError:
                return "Error: DOCX support not installed. Install python-docx package."
                
        elif full_path.suffix in ['.xlsx']:
            # For Excel, we'll need openpyxl
            try:
                import openpyxl
                wb = openpyxl.load_workbook(full_path)
                text = f"Excel file with {len(wb.sheetnames)} sheets: {', '.join(wb.sheetnames)}\n\n"
                for sheet_name in wb.sheetnames[:5]:  # First 5 sheets
                    ws = wb[sheet_name]
                    text += f"\n=== Sheet: {sheet_name} ===\n"
                    for row in list(ws.rows)[:100]:  # First 100 rows
                        text += "\t".join([str(cell.value) if cell.value is not None else "" for cell in row]) + "\n"
                content = text[:max_chars]
            except ImportError:
                return "Error: XLSX support not installed. Install openpyxl package."
        else:
            # Text-based files
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)
        
        result = {
            'file_path': file_path,
            'size': file_size,
            'extension': full_path.suffix,
            'content': content,
            'truncated': len(content) >= max_chars
        }
        
        import json
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error reading document {file_path}: {e}", exc_info=True)
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
    try:
        if not query:
            return "Error: Search query cannot be empty"
        
        search_query = query if case_sensitive else query.lower()
        matches = []
        
        for file_path in DOCUMENTS_PATH.rglob('*'):
            if not file_path.is_file():
                continue
                
            if file_path.suffix not in ALLOWED_EXTENSIONS:
                continue
                
            if file_extension and file_path.suffix != file_extension:
                continue
            
            try:
                # Only search text-based files for now
                if file_path.suffix in ['.txt', '.md', '.json', '.yaml', '.yml', '.log', '.csv']:
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
            except Exception as e:
                logger.debug(f"Error searching {file_path}: {e}")
                continue
        
        import json
        return json.dumps({
            'query': query,
            'total_matches': len(matches),
            'matches': matches[:50]  # Limit to 50 results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}", exc_info=True)
        return f"Error searching documents: {str(e)}"


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
    # Ensure documents directory exists
    DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting MCP Document Server")
    logger.info(f"Serving documents from: {DOCUMENTS_PATH}")
    
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
        mcp.run(transport='sse', host=host, port=port)


if __name__ == '__main__':
    main()
