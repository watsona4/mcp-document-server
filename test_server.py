#!/usr/bin/env python3
"""
Test script for MCP Document Server
"""

import asyncio
import json
import sys
from pathlib import Path

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("Error: MCP SDK not installed")
    print("Install with: pip install mcp")
    sys.exit(1)


async def test_mcp_server():
    """Test the MCP document server"""
    
    print("=" * 50)
    print("MCP Document Server Test")
    print("=" * 50)
    print()
    
    # Create server parameters
    server_params = StdioServerParameters(
        command="docker",
        args=["exec", "-i", "mcp-document-server", "python", "server.py"],
        env={"MCP_TRANSPORT": "stdio"}
    )
    
    print("📡 Connecting to MCP server...")
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                print("✅ Connected successfully!")
                print()
                
                # List available tools
                print("🔧 Available Tools:")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")
                print()
                
                # Test list_documents
                print("📁 Testing list_documents...")
                result = await session.call_tool("list_documents", {
                    "subdirectory": "",
                    "recursive": False
                })
                
                if result.content:
                    data = json.loads(result.content[0].text)
                    print(f"  Found {data['total_files']} documents")
                    if data['documents']:
                        print("  Sample documents:")
                        for doc in data['documents'][:3]:
                            print(f"    - {doc['name']} ({doc['size']} bytes)")
                print()
                
                # Test search (if there are documents)
                print("🔍 Testing search_documents...")
                result = await session.call_tool("search_documents", {
                    "query": "test",
                    "case_sensitive": False
                })
                
                if result.content:
                    data = json.loads(result.content[0].text)
                    print(f"  Found {data['total_matches']} matches")
                print()
                
                print("=" * 50)
                print("✅ All tests passed!")
                print("=" * 50)
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_http_endpoint():
    """Test the HTTP/SSE endpoint"""
    import requests
    
    print()
    print("=" * 50)
    print("Testing HTTP Endpoint")
    print("=" * 50)
    print()
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ HTTP endpoint is healthy")
        else:
            print(f"⚠️  HTTP endpoint returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to HTTP endpoint: {e}")
        print("   Make sure the server is running with: docker-compose up -d")


if __name__ == "__main__":
    # Test HTTP endpoint first
    test_http_endpoint()
    
    # Test MCP protocol
    print()
    asyncio.run(test_mcp_server())
