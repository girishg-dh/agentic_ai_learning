#!/bin/bash

# Agentic AI Learning Environment Setup Script
# For experienced ML Engineers starting Agentic AI journey
# Compatible with Linux/macOS. For Windows, use Git Bash or WSL.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="agentic_ai_learning"
PYTHON_VERSION="3.11"
VENV_NAME="agentic_ai_env"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed. Please install it first."
        return 1
    fi
    return 0
}

# Header
echo "🤖 Agentic AI Development Environment Setup"
echo "========================================"
echo "Setting up professional environment for Agentic AI learning"
echo ""

# Check prerequisites
print_status "Checking prerequisites..."
check_command "python3" || exit 1
check_command "git" || exit 1
check_command "curl" || print_warning "curl not found - some downloads may fail"

# Check for uv, install if missing
if ! command -v uv &> /dev/null; then
    print_warning "uv not found. Installing uv for faster package management..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Verify uv installation
    if ! command -v uv &> /dev/null; then
        print_warning "uv installation failed. Falling back to pip."
        USE_UV=false
    else
        print_success "uv installed successfully"
        USE_UV=true
    fi
else
    print_success "uv found"
    USE_UV=true
fi

# Check Python version
PYTHON_VER=$(python3 --version | cut -d" " -f2)
PYTHON_MAJOR=$(echo $PYTHON_VER | cut -d"." -f1)
PYTHON_MINOR=$(echo $PYTHON_VER | cut -d"." -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 9 ]]; then
    print_error "Python 3.9+ required. Found: $PYTHON_VER"
    exit 1
fi
print_success "Python $PYTHON_VER detected"

# Create project directory
print_status "Creating project structure..."
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Create virtual environment using uv or python
print_status "Creating virtual environment..."
if [[ "$USE_UV" == true ]]; then
    uv venv $VENV_NAME --python python3
    print_success "Virtual environment created with uv"
else
    python3 -m venv $VENV_NAME
    print_success "Virtual environment created with python venv"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Verify virtual environment is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi
print_success "Virtual environment activated: $VIRTUAL_ENV"

# Function to install packages
install_packages() {
    local description="$1"
    shift
    local packages=("$@")
    
    print_status "Installing $description..."
    
    if [[ "$USE_UV" == true ]]; then
        # uv can install multiple packages in one command more efficiently
        uv pip install "${packages[@]}"
    else
        # Fallback to pip with optimizations
        pip install --no-cache-dir "${packages[@]}"
    fi
}

# Upgrade pip/uv
if [[ "$USE_UV" == true ]]; then
    print_status "uv is self-updating, ensuring latest version..."
    # uv updates itself, but we can sync the environment
    uv pip install --upgrade pip setuptools wheel
else
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
fi

# Core AI/ML Libraries
install_packages "core AI libraries" \
    "openai>=1.30.0" \
    "anthropic>=0.25.0" \
    "python-dotenv>=1.0.0"

# LangChain Ecosystem
install_packages "LangChain ecosystem" \
    "langchain>=0.2.0" \
    "langchain-openai>=0.1.0" \
    "langchain-community>=0.2.0" \
    "langgraph>=0.1.0" \
    "langsmith>=0.1.0"

# LlamaIndex
install_packages "LlamaIndex" \
    "llama-index>=0.10.50" \
    "llama-index-llms-openai>=0.1.20" \
    "llama-index-embeddings-openai>=0.1.10"

# Multi-Agent Frameworks
install_packages "multi-agent frameworks" \
    "crewai>=0.30.0" \
    "crewai-tools>=0.4.0" \
    "pyautogen>=0.2.30"

# Vector Databases
install_packages "vector databases" \
    "chromadb>=0.5.0" \
    "weaviate-client>=4.6.0" \
    "qdrant-client>=1.9.0"

# Web Tools & APIs
install_packages "web tools" \
    "requests>=2.31.0" \
    "beautifulsoup4>=4.12.0" \
    "httpx>=0.27.0" \
    "aiohttp>=3.9.0"

# Development Tools
install_packages "development tools" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "streamlit>=1.35.0" \
    "chainlit>=1.1.0" \
    "jupyter>=1.0.0" \
    "ipykernel>=6.29.0"

# Data Processing
install_packages "data processing libraries" \
    "pandas>=2.2.0" \
    "numpy>=1.26.0" \
    "matplotlib>=3.8.0" \
    "plotly>=5.20.0"

# Document Processing
install_packages "document processing" \
    "pypdf>=4.2.0" \
    "python-docx>=1.1.0" \
    "tiktoken>=0.7.0"

# Evaluation & Safety
install_packages "evaluation tools" \
    "guardrails-ai>=0.5.0" \
    "ragas>=0.1.10"

# Utilities
install_packages "utilities" \
    "rich>=13.7.0" \
    "typer>=0.12.0" \
    "pydantic>=2.7.0"

# Generate requirements.txt
if [[ "$USE_UV" == true ]]; then
    uv pip freeze > requirements.txt
else
    pip freeze > requirements.txt
fi
print_success "Generated requirements.txt"

# Create project structure
print_status "Creating project structure..."
mkdir -p {week_01_foundations,week_02_tools,week_03_rag_memory,week_04_reasoning,week_05_multi_agent,week_06_production,week_07_deployment,week_08_capstone}/{notebooks,scripts,data}
mkdir -p shared/{agents,tools,evaluators,config}
mkdir -p {notebooks,tests,docs,deployments}
mkdir -p deployments/{docker,kubernetes,cloud}

# Create __init__.py files
find . -type d -name "*.py" -exec touch {}/__init__.py \;
touch shared/__init__.py
touch shared/agents/__init__.py
touch shared/tools/__init__.py
touch shared/evaluators/__init__.py
touch shared/config/__init__.py

# Create .env template
print_status "Creating configuration files..."
cat > .env.template << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_org_id_here

# Anthropic (Claude) Configuration  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LangSmith (Monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=agentic-ai-learning

# Vector Databases (Optional)
PINECONE_API_KEY=your_pinecone_key_here
WEAVIATE_URL=your_weaviate_url_here
QDRANT_URL=your_qdrant_url_here

# Search APIs (Optional)
SERPAPI_API_KEY=your_serpapi_key_here
TAVILY_API_KEY=your_tavily_key_here

# Other Services
HUGGINGFACE_API_KEY=your_hf_key_here
WANDB_API_KEY=your_wandb_key_here
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
agentic_ai_env/
venv/
ENV/

# Environment variables
.env
.env.local
.env.development
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints

# Data
*.csv
*.json
*.parquet
data/
datasets/
models/
*.pkl
*.joblib

# Logs
logs/
*.log
.cache/
EOF

# Create VS Code settings
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./agentic_ai_env/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "jupyter.askForKernelRestart": false,
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000
}
EOF

cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.flake8",
        "GitHub.copilot",
        "GitHub.copilot-chat"
    ]
}
EOF

# Create setup validation script
cat > validate_setup.py << 'EOF'
#!/usr/bin/env python3
import sys
import importlib
import os
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    packages = [
        ('openai', 'OpenAI API'),
        ('langchain', 'LangChain'),
        ('chromadb', 'ChromaDB'),
        ('crewai', 'CrewAI'),
        ('llama_index', 'LlamaIndex'),
        ('streamlit', 'Streamlit'),
        ('fastapi', 'FastAPI'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests')
    ]
    
    failed = []
    for package, name in packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            failed.append(package)
    
    return len(failed) == 0

def test_environment():
    """Test environment setup"""
    checks = [
        ('.env.template', 'Environment template'),
        ('requirements.txt', 'Requirements file'),
        ('.gitignore', 'Git ignore'),
        ('.vscode/settings.json', 'VS Code settings')
    ]
    
    all_good = True
    for file_path, description in checks:
        if Path(file_path).exists():
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            all_good = False
    
    return all_good

def main():
    print("🧪 Validating Agentic AI Environment Setup\n")
    
    print("📦 Package Imports:")
    imports_ok = test_imports()
    
    print("\n📁 Project Structure:")
    env_ok = test_environment()
    
    if imports_ok and env_ok:
        print("\n🎉 Setup validation successful!")
        print("\n📋 Next Steps:")
        print("1. Copy .env.template to .env")
        print("2. Add your API keys to .env file")
        print("3. Run: source agentic_ai_env/bin/activate")
        print("4. Start with Week 1 of the learning plan")
        return 0
    else:
        print("\n❌ Setup validation failed!")
        print("Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x validate_setup.py

# Create README
cat > README.md << 'EOF'
# Agentic AI Learning Environment

Professional development environment for learning Agentic AI Engineering.

## Quick Start

1. **Activate Environment:**
   ```bash
   source agentic_ai_env/bin/activate
   ```

2. **Configure API Keys:**
   ```bash
   cp .env.template .env
   # Edit .env with your actual API keys
   ```

3. **Validate Setup:**
   ```bash
   python validate_setup.py
   ```

4. **Open in VS Code:**
   ```bash
   code .
   ```

## Project Structure

- `week_XX_*/` - Weekly learning projects
- `shared/` - Reusable utilities and agents
- `notebooks/` - Jupyter exploration notebooks
- `deployments/` - Production deployment configs

## Learning Plan

Follow the 8-week structured learning plan for Agentic AI mastery.

## Environment Management

- Virtual environment: `agentic_ai_env/`
- Dependencies: `requirements.txt`
- Configuration: `.env` (copy from `.env.template`)

Happy Learning! 🤖
EOF

# Initialize git repository
print_status "Initializing git repository..."
git init
git add .
git commit -m "Initial setup: Agentic AI learning environment"

# Final validation
print_status "Running setup validation..."
python validate_setup.py

print_success "Setup complete!"
echo ""
echo "🎯 Quick Start Commands:"
echo "1. Activate environment: source $VENV_NAME/bin/activate"
echo "2. Configure API keys: cp .env.template .env && nano .env"
echo "3. Open VS Code: code ."
echo "4. Start learning: Begin with Week 1!"
echo ""
if [[ "$USE_UV" == true ]]; then
    echo "💨 Using uv for fast package management!"
    echo "   Add packages: uv pip install <package>"
    echo "   Update packages: uv pip install --upgrade <package>"
else
    echo "📦 Using pip for package management"
    echo "   Add packages: pip install <package>"
fi
echo ""
echo "📁 Project location: $(pwd)"
echo "🐍 Python interpreter: $(pwd)/$VENV_NAME/bin/python"