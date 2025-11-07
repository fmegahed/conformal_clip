# Contributing to conformal_clip

Thank you for your interest in contributing to conformal_clip! We welcome contributions from the community.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/fmegahed/conformal_clip.git
   cd conformal_clip
   ```

2. **Install OpenAI CLIP** (required dependency):
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

4. **Install development dependencies** (optional):
   ```bash
   pip install -e ".[data]"  # Include example dataset
   ```

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Add comprehensive docstrings to all public functions using NumPy style
- Use type hints for function arguments and return values
- Keep functions focused and modular

## Documentation

- Update docstrings when modifying function signatures or behavior
- Add examples to docstrings when appropriate
- Update README.md if adding new features or changing usage patterns
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/) format

## Testing

- Add tests for new functionality in the `tests/` directory
- Ensure existing tests pass before submitting changes
- Run tests with: `pytest tests/`

## Submitting Changes

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** with clear, descriptive commits
4. **Test your changes** thoroughly
5. **Update documentation** as needed
6. **Submit a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

## Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Include a clear description and minimal reproducible example
- Specify your Python version, OS, and relevant package versions

## Code of Conduct

- Be respectful and inclusive in all interactions
- Provide constructive feedback
- Focus on what is best for the community

## Questions?

Feel free to open an issue for questions or reach out to the maintainers:
- Fadel M. Megahed (fmegahed@miamioh.edu)
- Ying-Ju (Tessa) Chen (ychen4@udayton.edu)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
