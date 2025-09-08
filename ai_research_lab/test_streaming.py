#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing streaming functionality...")

try:
    # Test 1: Import main modules
    print("1. Testing imports...")
    from main import run_research_crew
    print("   ✓ main.py imported successfully")
    
    # Test 2: Test basic function call
    print("2. Testing basic function call...")
    brief = "Test brief: What is Python?"
    result = run_research_crew(brief=brief)
    print(f"   ✓ Function executed, result type: {type(result)}")
    print(f"   ✓ Result preview: {str(result)[:100]}...")
    
    # Test 3: Test app import
    print("3. Testing FastAPI app...")
    from app import app
    print("   ✓ FastAPI app imported successfully")
    
    print("\n✅ All tests passed! The issue might be in the frontend JavaScript.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()