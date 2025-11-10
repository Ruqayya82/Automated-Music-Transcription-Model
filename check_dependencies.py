#!/usr/bin/env python
"""
Dependency Checker for TranscribeAI

Run this script to verify all required packages are installed correctly.
Usage: python check_dependencies.py
"""

import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("=" * 60)
    print("TranscribeAI - Dependency Checker")
    print("=" * 60)
    print()
    
    missing_packages = []
    optional_packages = []
    
    # Required packages
    required = {
        'flask': 'Flask',
        'flask_cors': 'flask-cors',
        'yaml': 'PyYAML',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'pretty_midi': 'pretty_midi',
        'music21': 'music21',
        'tqdm': 'tqdm',
    }
    
    # Optional packages
    optional = {
        'torch': 'torch (PyTorch)',
        'sklearn': 'scikit-learn',
        'jupyter': 'jupyter',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
    }
    
    print("Checking Required Packages:")
    print("-" * 60)
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package:<30} INSTALLED")
        except ImportError:
            print(f"✗ {package:<30} MISSING")
            missing_packages.append(package)
    
    print()
    print("Checking Optional Packages:")
    print("-" * 60)
    
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"✓ {package:<30} INSTALLED")
        except ImportError:
            print(f"○ {package:<30} NOT INSTALLED (optional)")
            optional_packages.append(package)
    
    print()
    print("=" * 60)
    
    if missing_packages:
        print("⚠️  MISSING REQUIRED PACKAGES:")
        print()
        print("Please install missing packages with:")
        print(f"  pip install {' '.join(missing_packages)}")
        print()
        print("Or install all dependencies with:")
        print("  pip install -r requirements.txt")
        print()
        return False
    else:
        print("✓ All required packages are installed!")
        print()
        if optional_packages:
            print(f"Note: {len(optional_packages)} optional packages not installed.")
            print("These are not required but may enhance functionality.")
        print()
        print("You're ready to run TranscribeAI!")
        print("Start the app with: python run.py")
        print()
        return True

if __name__ == '__main__':
    success = check_dependencies()
    sys.exit(0 if success else 1)
