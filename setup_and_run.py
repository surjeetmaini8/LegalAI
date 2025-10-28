"""Setup and run script for Legal AI Chat Assistant"""

import subprocess
import sys
import os
from pathlib import Path



def create_directories():
    """Create necessary directories"""
    dirs = [
        'data',
        'data/documents',
        'data/vector_store',
        'modules',
        'utils'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def install_dependencies():
    
    print(" Installing dependencies...")
    print("This may take several minutes on first run\n")
    
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            "requirements.txt"
        ])
        print("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error installing dependencies: {e}")
        return False

def create_env_file():
    if not Path('.env').exists() and Path('.env.example').exists():
        import shutil
        shutil.copy('.env.example', '.env')
        print("Created .env file from template")
    else:
        print(".env file already exists or template not found")

def run_streamlit():
    
    print("Starting Legal AI Chat Assistant...")
    
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port=8501",
            "--server.headless=false"
        ])
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f" Error running application: {e}")

def main():
    
    print("Creating project directories...")
    create_directories()
    
    print("Setting up configuration...")
    create_env_file()
    
    install = input("Install/update dependencies? (y/n) [y]: ").strip().lower()
    
    if install != 'n':
        if not install_dependencies():
            print("Setup failed. Please check errors above.")
            sys.exit(1)
    else:
        print("Skipping dependency installation")
    
    run_app = input("Run the application now? (y/n) [y]: ").strip().lower()
    
    if run_app != 'n':
        run_streamlit()
    else:
        print("Setup complete!")


if __name__ == "__main__":
    main()
