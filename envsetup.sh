#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11.0"
PROJECT_DIR="$HOME/thesis_project"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Function to install pyenv if not present
install_pyenv() {
    if [ ! -d "$HOME/.pyenv" ]; then
        log "Installing pyenv..."
        curl -sSL https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
        
        # Add to current session
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
        
        # Add to shell profile for persistence
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    else
        log "pyenv already installed"
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    fi
}

# Function to install Poetry if not present
install_poetry() {
    if ! command -v poetry &> /dev/null; then
        log "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    else
        log "Poetry already installed"
        export PATH="$HOME/.local/bin:$PATH"
    fi
}

# Function to setup Python environment
setup_python() {
    log "Setting up Python $PYTHON_VERSION..."
    
    # Check if Python version is already installed
    if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
        log "Installing Python $PYTHON_VERSION..."
        pyenv install "$PYTHON_VERSION"
    else
        log "Python $PYTHON_VERSION already installed"
    fi
    
    # Set global Python version
    pyenv global "$PYTHON_VERSION"
    
    # Verify installation
    python_path=$(which python)
    python_version=$(python --version)
    log "Using Python: $python_path ($python_version)"
}

# Function to setup project environment
setup_project() {
    log "Setting up project environment..."
    
    # Create project directory if it doesn't exist
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    
    # Download/sync your project files
    # Option 1: If you have a git repository
    if [ ! -d ".git" ]; then
        log "Cloning project repository..."
        git clone https://github.com/yourusername/your-thesis-repo.git .
    else
        log "Updating project repository..."
        git pull origin main
    fi
    
    # Option 2: If you upload files manually (uncomment if needed)
    # log "Please upload your Python files and pyproject.toml to $PROJECT_DIR"
    # read -p "Press enter when files are uploaded..."
    
    # Install dependencies with Poetry
    if [ -f "pyproject.toml" ]; then
        log "Installing project dependencies..."
        poetry install
    else
        error "pyproject.toml not found in $PROJECT_DIR"
    fi
}

# Function to run script with arguments
run_script() {
    local script_name="$1"
    shift # Remove script name from arguments
    local script_args="$@"
    
    cd "$PROJECT_DIR"
    
    if [ -f "$script_name" ]; then
        log "Running: $script_name $script_args"
        poetry run python "$script_name" $script_args
    else
        error "Script $script_name not found in $PROJECT_DIR"
    fi
}

# Main execution
main() {
    log "Starting thesis environment setup..."
    
    # Install dependencies
    install_pyenv
    install_poetry
    setup_python
    setup_project
    
    log "Environment setup complete! You can now run:"
    echo "  poetry run python script1.py arg1 arg2 arg3"
    echo "  poetry run python script2.py different args"
    echo ""
    echo "Available scripts:"
    ls -1 *.py 2>/dev/null || echo "  No Python scripts found"

}

