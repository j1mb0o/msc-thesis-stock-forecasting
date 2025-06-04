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

install_build_dep(){
	sudo apt update 
	sudo apt install make build-essential libssl-dev zlib1g-dev \
	libbz2-dev libreadline-dev libsqlite3-dev curl git \
	libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
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
    install_build_dep
    setup_python
    
    log "Environment setup complete! You can now run:"
    echo "  poetry run python script1.py arg1 arg2 arg3"
    echo "  poetry run python script2.py different args"
    echo ""
    echo "Available scripts:"
    ls -1 *.py 2>/dev/null || echo "  No Python scripts found"

}

main
