# Contributing to MCP Document Server

Thanks for your interest in contributing! This project was built for personal use but contributions are welcome.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone git@github.com:yourusername/mcp-document-server.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes: `python test_server.py`
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio black flake8

# Run tests
pytest tests/

# Format code
black server.py

# Lint code
flake8 server.py
```

## Code Style

- Follow PEP 8
- Use Black for formatting
- Add docstrings to functions
- Keep functions focused and small
- Write tests for new features

## Testing

Before submitting a PR:

1. Run all tests: `pytest tests/`
2. Test Docker build: `docker-compose build`
3. Test with Claude Desktop if possible
4. Update documentation if needed

## Adding New File Type Support

To add support for a new file type:

1. Add the extension to `ALLOWED_EXTENSIONS` default
2. Add parsing logic in `read_document()` function
3. Update README.md with the new file type
4. Test with sample files

Example:

```python
elif full_path.suffix == '.newtype':
    try:
        import newtype_parser
        content = newtype_parser.parse(full_path)
    except ImportError:
        return "Error: newtype support not installed"
```

## Documentation

- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Keep documentation clear and concise

## Reporting Issues

When reporting issues, please include:

- OS and version
- Docker version (if applicable)
- Python version
- Error messages and logs
- Steps to reproduce
- Expected vs actual behavior

## Feature Requests

Feature requests are welcome! Please:

- Check if it already exists in Issues
- Describe the use case
- Explain why it would be useful
- Be open to discussion

## Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md (if present)
5. Reference any related issues
6. Wait for review

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the code, not the person
- Assume good intentions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue or discussion on GitHub!

---

Thanks for contributing! 🎉
