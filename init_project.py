"""
./init_project.py
Simple template-to-project transformer

Converts template placeholders to actual project values and cleans up template artifacts.
Designed for simplicity and reliability - one execution, no configuration files needed.
"""

import json
import re
from pathlib import Path
import sys
import subprocess

def main():
    """Transform template into personalized project."""
    
    print("🚀 Initializing project from template")
    print("=" * 40)
    
    # Get the essential information we need
    project_info = gather_project_info()
    
    # Show what we're about to do
    print(f"\n📋 Setting up project: {project_info['project_name']}")
    print(f"   Python version: {project_info['python_version']}")

    # Confirm before proceeding
    if input("\nProceed with initialization? [Y/n]: ").lower().startswith('n'):
        print("Initialization cancelled.")
        return
    
    # Do the transformation
    # process_template_files(project_info)
    # rename_source_directory(project_info)
    # cleanup_template_files()

    print("\n🔄 Since this step should happen in a codespace, we will install UV using pipx.")
    print("   This will ensure that the project runs smoothly in the cloud environment.")
    print("   Locally, you can choose to install uv with curl or in a virtual environment with pip: https://docs.astral.sh/uv/getting-started/installation/")

    install_uv()
    initialize_project(project_info)
    install_development_dependencies(project_info)
    cleanup_unnecessary_files()

    print(f"\n✅ Project '{project_info['project_name']}' initialized successfully!")
    print("\nNext steps:")
    print("  1. Run 'make setup' to install dependencies")
    print("  2. Run 'make test' to verify everything works")
    print("  3. Start coding! 🚀")


def gather_project_info():
    """Get the essential project information from user."""
    
    # Get project name with validation
    original_dir = Path.cwd().name.lower()
    current_dir = Path.cwd().name.lower().replace("-", "_")
    #given that this repo is a template, the user selects the name when using the template
    project_name = current_dir

    # User must specify python version, with default 3.11:
    python_version = input("Python version [3.11]: ").strip() or "3.11"
    install_development_dependencies = input("Install development dependencies? [Y/n]: ").strip().lower() != 'n'

    # Validate project name (must be valid Python identifier)
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', project_name):
        print("❌ Project name must be a valid Python identifier")
        print("   (letters, numbers, underscores only; cannot start with number)")
        sys.exit(1)
    
    # Get other essential info    
    return {
        'project_name': project_name,
        'python_version': python_version,
        'original_dir': original_dir,
        'install_dev_dependencies': install_development_dependencies,
    }

def install_uv():
    """Install UV in the system."""
    print("🔄 Installing UV...")
    if subprocess.call(["which", "uv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) != 0:
        subprocess.call(["pipx", "install", "uv"])
    
    # Check if UV is installed
    if subprocess.call(["which", "uv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
        print("✅ UV installed successfully.")
    else:
        print("❌ UV installation failed.")

def update_toml_file(project_info):
    # Create venv
    subprocess.call([
        "python",
        "-m",
        "venv",
        ".venv"
        ])
    
    # Activate venv
    subprocess.call(". .venv/bin/activate", shell=True)

    # Install toml
    subprocess.call(["pip", "install", "toml"])

    import toml

    # Read the file
    with open('pyproject.toml', 'r') as f:
        data = toml.load(f)

    # Modify content
    # Remove the old entry
    if project_info['original_dir'] in data['project']['scripts']:
        del data['project']['scripts'][project_info['original_dir']]

    # Add the new entry
    data['project']['scripts'][project_info['original_dir']] = f"{project_info['project_name']}.main:main"

    # Write back
    with open('pyproject.toml', 'w') as f:
        toml.dump(data, f)

def create_dummy_main_script(project_info):
    """Create a dummy main script."""
    print("\n🔄 Creating dummy main script...")
    main_script = Path(f"src/{project_info['project_name']}/main.py")
    project_name = project_info['project_name']
    main_script.write_text(
        f'def main():\n    print(f"{project_name} is ready!")\n\nif __name__ == "__main__":\n    main()'
    )
    print(f"   Created dummy main script: {main_script}")
    
def initialize_project(project_info):
    """Initialize the project directory structure using uv."""
    print("\n🔄 Initializing project structure...")

    subprocess.call([
        "uv",
        "init",
        "--package",
        "--python",
        project_info['python_version']
    ])

    #Verify if the folder that was created inside the src folder has the project name:
    src_folder = Path("src")
    if src_folder.exists():
        project_folder = src_folder / project_info['project_name']
        if project_folder.exists():
            print(f"   Verified project folder: {project_folder}")
            
            # Clean the __init__ file in the project folder to remove main function:
            init_file = project_folder / "__init__.py"
            if init_file.exists():
                # Remove all the contents:
                init_file.write_text("")
                print(f"   Cleaned __init__ file: {init_file}")

                create_dummy_main_script(project_info)

                update_toml_file(project_info)

        else:
            print(f"❌ Project folder not found: {project_folder}")

    # Update Pyright configuration with correct source path
    pyright_config = Path("pyrightconfig.json")
    if pyright_config.exists():
        config = json.loads(pyright_config.read_text())
        config["include"] = [f"src/{project_info['project_name']}", "tests"]
        pyright_config.write_text(json.dumps(config, indent=2))
        print("   Updated Pyright configuration")

    # Make tests directory:
    tests_folder = Path("tests")
    if not tests_folder.exists():
        tests_folder.mkdir()
        print(f"   Created tests directory: {tests_folder}")

    # Create a __init__.py file in the tests directory
    init_file = tests_folder / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"   Created test package: {init_file}")

    # sample uv run:
    subprocess.call([
        "uv",
        "run",
        f"{project_info['original_dir']}"
    ])

def install_development_dependencies(project_info):

    if project_info['install_dev_dependencies']:
        print("\n🔄 Installing development dependencies...")

        subprocess.call([
            "uv",
            "add",
            "--dev",
            "ruff",
            "pyright",
            "pytest-cov",
            "coverage[toml]",
            "pre-commit"
        ])
    else:
        print("\n🔄 Skipping development dependencies installation.")

def cleanup_unnecessary_files():

    # Remove the .venv directory if it exists
    venv_dir = Path(".venv")
    if venv_dir.exists():
        subprocess.call(["rm", "-rf", venv_dir])
        print(f"   Removed virtual environment: {venv_dir}")

    # Remove the __pycache__ directories
    for pycache in Path("src").rglob("__pycache__"):
        subprocess.call(["rm", "-rf", pycache])
        print(f"   Removed __pycache__ directory: {pycache}")

    # Remove the .ruff_cache directory if it exists
    ruff_cache = Path(".ruff_cache")
    if ruff_cache.exists():
        subprocess.call(["rm", "-rf", ruff_cache])
        print(f"   Removed Ruff cache directory: {ruff_cache}")
    
    # README file
    readme = Path("README.md")
    if readme.exists():
        subprocess.call(["rm", readme])

    # Remove this script
    script_file = Path(__file__)
    if script_file.exists():
        subprocess.call(["rm", "-rf", script_file])
        print(f"   Removed init script file: {script_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInitialization cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)