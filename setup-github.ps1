# MCP Document Server - GitHub Setup (PowerShell)
# Run this script on Windows to push to GitHub

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "MCP Document Server - GitHub Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: git is not installed" -ForegroundColor Red
    Write-Host "Please install git first: https://git-scm.com/download/win"
    exit 1
}

# Check if GitHub CLI is installed
$hasGhCli = Get-Command gh -ErrorAction SilentlyContinue
if ($hasGhCli) {
    Write-Host "✅ GitHub CLI detected" -ForegroundColor Green
} else {
    Write-Host "⚠️  GitHub CLI not found (optional)" -ForegroundColor Yellow
    Write-Host "Install from: https://cli.github.com/" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "This script will:"
Write-Host "1. Initialize a git repository"
Write-Host "2. Create initial commit"
Write-Host "3. Push to GitHub"
Write-Host ""

# Get repository details
$repoName = Read-Host "Repository name [mcp-document-server]"
if ([string]::IsNullOrWhiteSpace($repoName)) {
    $repoName = "mcp-document-server"
}

$githubUsername = Read-Host "Your GitHub username"
if ([string]::IsNullOrWhiteSpace($githubUsername)) {
    Write-Host "❌ GitHub username is required" -ForegroundColor Red
    exit 1
}

$visibility = Read-Host "Make repository private? (Y/n)"
if ($visibility -match "^[Nn]$") {
    $isPrivate = $false
    $visibilityText = "public"
} else {
    $isPrivate = $true
    $visibilityText = "private"
}

Write-Host ""
Write-Host "Repository Details:" -ForegroundColor Yellow
Write-Host "  Name: $repoName"
Write-Host "  Owner: $githubUsername"
Write-Host "  Visibility: $visibilityText"
Write-Host ""

$confirm = Read-Host "Continue? (y/N)"
if ($confirm -notmatch "^[Yy]$") {
    Write-Host "Aborted."
    exit 0
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Setting up Git repository..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Initialize git repository
if (-not (Test-Path .git)) {
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "⚠️  Git repository already exists" -ForegroundColor Yellow
}

# Add all files
git add .

# Check if there are changes to commit
$statusOutput = git status --porcelain
if ([string]::IsNullOrWhiteSpace($statusOutput)) {
    Write-Host "⚠️  No changes to commit" -ForegroundColor Yellow
} else {
    # Create initial commit
    $commitMessage = @"
Initial commit: MCP Document Server

- Full MCP server implementation with FastMCP
- Docker and docker-compose setup
- Support for PDF, DOCX, XLSX, and text files
- Integration guides for Claude Desktop, OPNsense, and Tailscale
- Read-only document access with security features
- Systemd service configuration
- Comprehensive documentation
"@
    
    git commit -m $commitMessage
    Write-Host "✅ Initial commit created" -ForegroundColor Green
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Creating GitHub repository..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

if ($hasGhCli) {
    # Use GitHub CLI to create repo
    Write-Host "Creating repository using GitHub CLI..."
    
    if ($isPrivate) {
        gh repo create $repoName --private --source=. --remote=origin --push
    } else {
        gh repo create $repoName --public --source=. --remote=origin --push
    }
    
    Write-Host "✅ Repository created and pushed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Repository URL: https://github.com/$githubUsername/$repoName"
    
} else {
    # Manual instructions
    Write-Host "GitHub CLI not available. Please follow these steps:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Go to: https://github.com/new"
    Write-Host ""
    Write-Host "2. Create a new repository with:"
    Write-Host "   - Repository name: $repoName"
    Write-Host "   - Visibility: $visibilityText"
    Write-Host "   - Do NOT initialize with README, .gitignore, or license"
    Write-Host ""
    Write-Host "3. After creating the repository, run these commands:"
    Write-Host ""
    Write-Host "   git remote add origin git@github.com:$githubUsername/$repoName.git"
    Write-Host "   git branch -M main"
    Write-Host "   git push -u origin main"
    Write-Host ""
    
    Read-Host "Press Enter after creating the repository on GitHub"
    
    # Add remote and push
    $remotes = git remote
    if ($remotes -contains "origin") {
        Write-Host "⚠️  Remote 'origin' already exists" -ForegroundColor Yellow
        $updateRemote = Read-Host "Update remote URL? (y/N)"
        if ($updateRemote -match "^[Yy]$") {
            git remote remove origin
            git remote add origin "git@github.com:$githubUsername/$repoName.git"
        }
    } else {
        git remote add origin "git@github.com:$githubUsername/$repoName.git"
    }
    
    # Rename to main branch if needed
    $currentBranch = git branch --show-current
    if ($currentBranch -ne "main") {
        git branch -M main
    }
    
    # Push to GitHub
    Write-Host ""
    Write-Host "Pushing to GitHub..."
    git push -u origin main
    
    Write-Host "✅ Repository pushed to GitHub!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Repository URL: https://github.com/$githubUsername/$repoName"
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host ""
Write-Host "1. View your repository:"
Write-Host "   https://github.com/$githubUsername/$repoName"
Write-Host ""
Write-Host "2. Clone on your home server:"
Write-Host "   git clone git@github.com:$githubUsername/$repoName.git"
Write-Host ""
Write-Host "3. Deploy the MCP server:"
Write-Host "   cd $repoName"
Write-Host "   ./install.sh"
Write-Host ""
Write-Host "🎉 Your MCP Document Server is now on GitHub!" -ForegroundColor Green
