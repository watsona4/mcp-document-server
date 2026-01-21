#!/bin/bash
set -e

echo "========================================="
echo "MCP Document Server Installation Script"
echo "========================================="
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed"
    echo "Please install Docker Compose first: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo "⚠️  Please edit .env to configure your server"
    echo ""
fi

# Create documents directory
if [ ! -d ./documents ]; then
    echo "📁 Creating documents directory..."
    mkdir -p ./documents
    echo "✅ Created ./documents directory"
    echo ""
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker-compose build

echo ""
echo "========================================="
echo "✅ Installation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit your configuration:"
echo "   nano .env"
echo ""
echo "2. Add your documents to:"
echo "   ./documents/"
echo ""
echo "3. Start the server:"
echo "   docker-compose up -d"
echo ""
echo "4. Check the logs:"
echo "   docker-compose logs -f"
echo ""
echo "5. Test the server:"
echo "   curl http://localhost:8000/health"
echo ""

# Optional: systemd installation
read -p "Would you like to install as a systemd service? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing systemd service..."
    
    # Get installation directory
    INSTALL_DIR=$(pwd)
    
    # Copy service file and update path
    sudo cp mcp-document-server.service /etc/systemd/system/
    sudo sed -i "s|/opt/mcp-document-server|$INSTALL_DIR|g" /etc/systemd/system/mcp-document-server.service
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable service
    sudo systemctl enable mcp-document-server.service
    
    echo "✅ Systemd service installed"
    echo ""
    echo "Systemd commands:"
    echo "  sudo systemctl start mcp-document-server    # Start service"
    echo "  sudo systemctl stop mcp-document-server     # Stop service"
    echo "  sudo systemctl status mcp-document-server   # Check status"
    echo "  sudo systemctl restart mcp-document-server  # Restart service"
    echo ""
fi

echo "========================================="
echo "🎉 Setup Complete!"
echo "========================================="
