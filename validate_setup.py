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
