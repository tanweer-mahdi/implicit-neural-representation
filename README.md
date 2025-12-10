# predelo-python-template

## Create the Environment

```bash
uv sync                     # installs everything listed in uv.lock
uv run pre-commit install   # installs the git hook to .git/hooks
```
Heads-up: The first uv sync takes ~20s; every run after that is near-instant.

## Usual Workflow

| Task                         | Command                             |
| ---------------------------- | ----------------------------------- |
| Run tests                    | `uv run pytest`                     |
| Lint & format the whole repo | `uv run pre-commit run --all-files` |
| Add a runtime dependency     | `uv add requests`                   |
| Add a dev-only tool          | `uv add --group lint ruff`          |
| Upgrade all packages         | `uv lock --upgrade`                 |
| Rebuild the virtualenv       | `rm -rf .venv && uv sync`           |

## Important files

| Path                      | Purpose                                                    |
| ------------------------- | ---------------------------------------------------------- |
| `pyproject.toml`          | Declares dependencies and tool (linter, formatter, typing) configs |
| `uv.lock`                 | Frozen versions & hashes — **never** hand-edit            |
| `.pre-commit-config.yaml` | Git hooks (`ruff`, `black`, `mypy`, `uv-lock`)            |
| `tests/`                  | `pytest` suite — with an example test                      |

## Type checking heads-up

```yaml
[tool.mypy]
python_version = "3.11"
strict = true
```
mypy analyzes the code as if it ships on 3.11. If you adopt 3.12-only syntax or APIs, bump the version in `pyproject.toml` and run:

```bash
uv lock --upgrade
```

## Troubleshooting

| Symptom                                  | Quick Fix                                                    |
| ---------------------------------------- | ------------------------------------------------------------ |
| `pre-commit failed: uv.lock out of date` | Run `uv lock` and recommit                                   |
| `mypy` rejects new 3.12 feature          | Raise `python_version` to `"3.12"`                           |
| Module X not found at runtime            | Run inside the env: `uv run python …` or activate with `source .venv/bin/activate` |
| First commit feels slow                  | Hook environments are being built; subsequent commits are fast |

