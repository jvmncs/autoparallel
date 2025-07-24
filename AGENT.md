# Agent Instructions

## Dependencies

This project uses the Astral stack for Python development:

- **uv** - Dependency management and Python environment management
- **ruff** - Linting and formatting
- **ty** - Type checking (pre-release)
- **pytest** - Unit testing
- **pytest-cov** - Coverage reporting for pytest

Install dependencies:
```bash
uv sync --group dev
```

## Commands

### Development
- **Install dependencies**: `uv sync --group dev`
- **Run Python script**: `uv run python <script>`
- **Activate shell**: `. .venv/bin/activate`

### Code Quality
- **Lint**: `uv run ruff check`
- **Format**: `uv run ruff format`
- **Lint + autofix**: `uv run ruff check --fix`
- **Type check**: `uv run ty check`

### Testing
- **Run tests**: `uv run pytest`
- **Run specific test**: `uv run pytest <test_file>`
- **Run with coverage**: `uv run pytest --cov`

## File Structure

```
/
├── src/                    # Main source code
├── spec/                   # Specifications and documentation
├── pyproject.toml         # Project configuration and dependencies
├── README.md              # Project documentation
└── AGENT.md               # This file
```

## Code Conventions

### Style Guide
- Follow **Google Python Style Guide** for naming conventions and code structure
- Use **snake_case** for variables, functions, and modules
- Use **PascalCase** for classes
- Use **UPPER_SNAKE_CASE** for constants

### Testing
- Test files should be named `*_test.py`, where `*` is the name of the source file being tested (Google style)
- Place test files alongside their corresponding source files
- Use descriptive test method names with `test_` prefix

### Formatting & Linting
- Code is automatically formatted with **ruff format**
- Linting rules are configured in `pyproject.toml`
- Line length: 88 characters
- Use double quotes for strings
- Use space indentation

### Imports
- Always use **absolute imports**: `from autoparallel.module.submodule import Class`
- Never use **relative imports**: `from .submodule import Class` or `from ..module import Class`
- Absolute imports avoid confusion and make dependencies clear
- Improves IDE support and refactoring capabilities

### Type Hints
- Use type hints for all function parameters and return values
- Type checking with **ty** (pre-release software)
- Follow PEP 484, 526, and 585 standards

## Environment Variables

None currently defined.

## Documentation Links

- [uv Documentation](https://docs.astral.sh/uv/)
- [ruff Documentation](https://docs.astral.sh/ruff/)
- [ty Documentation](https://github.com/astral-sh/ty/)
- [pytest Documentation](https://docs.pytest.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## Version Control

This project uses **jj** (Jujutsu) instead of Git for version control.

### Essential jj Commands

**Basic Workflow:**
```bash
jj st --no-pager                    # Check repository status
jj new                              # Make new commit for future changes; if changes already in working copy, skip this
jj describe -m "commit message"     # Add description to current commit
jj log --no-pager                   # View commit history
```

**Working with Remotes:**
```bash
jj git fetch --no-pager             # Fetch changes from remote
jj git push --no-pager               # Push current branch to remote
jj git push -c @ --no-pager          # Create auto-named branch and push current commit
```

**Common Patterns:**
```bash
jj new trunk                        # Start new work from trunk/main
jj squash -i                        # Interactive squash to organize changes
jj undo                             # Undo last operation if mistake made
```

### Key Concepts for Agents

- **Anonymous branches**: Commits are branches - no need to manage branch names
- **Always use `--no-pager`**: Prevents interactive TUIs that block automation
- **Two workflows**: Squash (build commits incrementally) or Edit (refine existing commits)
- **Recovery**: Use `jj undo` to revert mistakes, `jj op log --no-pager` to see operation history

### Commit Pattern for Agents

1. Make changes to files
2. `jj new` - Create commit with changes
3. `jj describe -m "descriptive commit message"` - Add description
4. `jj git push --no-pager` - Push to remote

### Common Pitfalls to Avoid

- **Never** use interactive commands without `--no-pager`
- **Don't** try to map Git concepts directly - jj uses different paradigms
- **Always** check `jj st --no-pager` before committing to see current state
- **Use** `jj undo` immediately if a command produces unexpected results
