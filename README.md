# MLops Project Template

This repository is a template designed to help you start Machine Learning and MLOps projects following software best practices. It includes tools to configure the environment, initialize the project structure, and enable continuous integration.

## Main Features

- **Guided initialization:** The [`init_project.py`](init_project.py) script transforms the template into a personalized project, prompting for relevant information and adapting the structure and configuration.
- **Automatic dependency setup:** Uses [UV](https://docs.astral.sh/uv/) to efficiently manage dependencies and virtual environments.
- **Modular structure:** Creates base folders and files for source code, tests, and scripts.
- **Integration with quality tools:** Includes configuration for linters ([Ruff](.ruff.toml)), type checking ([Pyright](pyrightconfig.json)), and pre-commit ([.pre-commit-config.yaml](.pre-commit-config.yaml)).
- **Makefile for common tasks:** Allows you to run formatting, linting, testing, and verification commands easily.

## Project Structure

```
<project-name>/
├── src/
│   └── <project_name>/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── __init__.py
├── init_project.py
├── .pre-commit-config.yaml
├── pyrightconfig.json
├── .ruff.toml
├── Makefile
├── README.md
└── .gitignore
```

## Getting Started

*This is the recommended usage:*

1. **Clone the repository as a template and open a codespace**  
   Use the "Use this template" option on GitHub to create your own repository, and open a Github codespace on main.

2. **Initialize the project**  
   Run the following command from the repository root to personalize the template:
   ```sh
   python init_project.py
   ```
   The script will ask for the Python version for now, but further customization will be added later.  
   UV will be installed and the project structure will be configured automatically.

3. **Start developing**  
   The main file will be at `src/<project_name>/main.py`.  
   You can add your modules and scripts following the created structure.

**After the project gets initialized, you can now use the setup either in codespaces, your local environment, or elsewhere, to have a CI setup with pre-commit hooks and use your code as a full package**

## IMPORTANT
- UV is a core dependency for both the setup and the subsequent code development added to the initialized project. You can use pip, pipx, or curl to install UV ([see here](https://docs.astral.sh/uv/)).

## Continuous Integration
Includes pre-commit configuration and can be easily integrated with CI/CD workflows. The most important aspects of CI are covered: formatting, linting, type checking, and testing. We make use of a Makefile for its portability.

In the future additional tasks and configurations may be added to further enhance the CI/CD capabilities, such as git workflows with Github Actions to enforce branch protection.

### Makefile Tasks

- `make format` — Format code with Ruff.
- `make lint` — Run Ruff linter.
- `make typecheck` — Type check with Pyright.
- `make test` — Run unit tests with Pytest.
- `security-check` - Security scan with pip-audit
- `make verify` — Run all the above tasks.

## Customization
The [`init_project.py`](init_project.py) script adapts the template to your project:
- Renames folders and files according to your project name.
- Updates Pyright and pyproject.toml configuration.
- Creates base files for code and tests.
