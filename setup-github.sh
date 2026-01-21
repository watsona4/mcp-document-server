#!/bin/bash
set -e

echo "========================================="
echo "MCP Document Server - GitHub Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Error: git is not installed${NC}"
    echo "Please install git first: sudo apt install git"
    exit 1
fi

# Check if gh CLI is installed (optional)
if command -v gh &> /dev/null; then
    HAS_GH_CLI=true
    echo -e "${GREEN}✅ GitHub CLI detected${NC}"
else
    HAS_GH_CLI=false
    echo -e "${YELLOW}⚠️  GitHub CLI not found (optional)${NC}"
fi

echo ""
echo "This script will:"
echo "1. Initialize a git repository"
echo "2. Create initial commit"
echo "3. Push to GitHub"
echo ""

# Get repository name
read -p "Repository name [mcp-document-server]: " REPO_NAME
REPO_NAME=${REPO_NAME:-mcp-document-server}

# Get GitHub username
read -p "Your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo -e "${RED}❌ GitHub username is required${NC}"
    exit 1
fi

# Repository visibility
read -p "Make repository private? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    VISIBILITY="public"
else
    VISIBILITY="private"
fi

echo ""
echo -e "${YELLOW}Repository Details:${NC}"
echo "  Name: $REPO_NAME"
echo "  Owner: $GITHUB_USERNAME"
echo "  Visibility: $VISIBILITY"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "========================================="
echo "Setting up Git repository..."
echo "========================================="
echo ""

# Initialize git repository
if [ ! -d .git ]; then
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}"
else
    echo -e "${YELLOW}⚠️  Git repository already exists${NC}"
fi

# Add all files
git add .

# Create initial commit
if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠️  No changes to commit${NC}"
else
    git commit -m "Initial commit: MCP Document Server

- Full MCP server implementation with FastMCP
- Docker and docker-compose setup
- Support for PDF, DOCX, XLSX, and text files
- Integration guides for Claude Desktop, OPNsense, and Tailscale
- Read-only document access with security features
- Systemd service configuration
- Comprehensive documentation"
    echo -e "${GREEN}✅ Initial commit created${NC}"
fi

echo ""
echo "========================================="
echo "Creating GitHub repository..."
echo "========================================="
echo ""

if [ "$HAS_GH_CLI" = true ]; then
    # Use GitHub CLI to create repo
    echo "Creating repository using GitHub CLI..."
    
    if [ "$VISIBILITY" = "private" ]; then
        gh repo create "$REPO_NAME" --private --source=. --remote=origin --push
    else
        gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
    fi
    
    echo -e "${GREEN}✅ Repository created and pushed!${NC}"
    echo ""
    echo "Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    
else
    # Manual instructions
    echo -e "${YELLOW}GitHub CLI not available. Please follow these steps:${NC}"
    echo ""
    echo "1. Go to: https://github.com/new"
    echo ""
    echo "2. Create a new repository with:"
    echo "   - Repository name: $REPO_NAME"
    echo "   - Visibility: $VISIBILITY"
    echo "   - Do NOT initialize with README, .gitignore, or license"
    echo ""
    echo "3. After creating the repository, run these commands:"
    echo ""
    echo "   git remote add origin git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    
    read -p "Press Enter after creating the repository on GitHub..."
    
    # Add remote and push
    if git remote | grep -q "origin"; then
        echo -e "${YELLOW}⚠️  Remote 'origin' already exists${NC}"
        read -p "Update remote URL? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git remote remove origin
            git remote add origin "git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
        fi
    else
        git remote add origin "git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
    fi
    
    # Rename to main branch if needed
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "$CURRENT_BRANCH" != "main" ]; then
        git branch -M main
    fi
    
    # Push to GitHub
    echo ""
    echo "Pushing to GitHub..."
    git push -u origin main
    
    echo -e "${GREEN}✅ Repository pushed to GitHub!${NC}"
    echo ""
    echo "Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
fi

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. View your repository:"
echo "   https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "2. Clone on your home server:"
echo "   git clone git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
echo ""
echo "3. Deploy the MCP server:"
echo "   cd $REPO_NAME"
echo "   ./install.sh"
echo ""
echo "🎉 Your MCP Document Server is now on GitHub!"
