"""
Environment setup module for Thai Isan Lute (Phin) Music Transcription

Handles dependency installation, GPU detection, and directory structure creation.
"""
import os
import sys
import torch
import subprocess
import importlib
from pathlib import Path


def check_gpu_availability():
    """
    Check if GPU is available and return GPU information.
    
    Returns:
        dict: Information about GPU availability and properties
    """
    gpu_info = {
        'available': torch.cuda.is_available(),
        'count': torch.cuda.device_count(),
        'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    }
    
    return gpu_info


def verify_dependencies():
    """
    Verify that all required dependencies are installed and accessible.
    
    Returns:
        dict: Status of each dependency
    """
    required_packages = [
        'librosa', 
        'pretty_midi', 
        'yt_dlp', 
        'torch', 
        'mir_eval', 
        'numpy', 
        'scipy'
    ]
    
    dependency_status = {}
    
    for package in required_packages:
        try:
            # Handle package names that differ from import names
            import_name = package.replace('-', '_')
            importlib.import_module(import_name)
            dependency_status[package] = True
        except ImportError:
            dependency_status[package] = False
    
    return dependency_status


def create_directories():
    """
    Create the required directory structure for the project.
    """
    directories = [
        'audio_sources',
        'sheet_music', 
        'output',
        'references',
        'tests/unit',
        'tests/integration',
        'tests/data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def install_dependencies():
    """
    Install required Python dependencies using pip.
    """
    required_packages = [
        'librosa',
        'pretty_midi',
        'yt-dlp',
        'torch',
        'mir_eval',
        'numpy',
        'scipy'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


def setup_environment():
    """
    Complete environment setup: install dependencies, create directories, check GPU.
    
    Returns:
        dict: Summary of setup results
    """
    print("Setting up Thai Isan Lute (Phin) Music Transcription Environment...")
    
    # Create directories
    print("Creating directory structure...")
    create_directories()
    
    # Install dependencies if needed
    print("Verifying dependencies...")
    deps_status = verify_dependencies()
    missing_deps = [pkg for pkg, status in deps_status.items() if not status]
    
    if missing_deps:
        print(f"Installing missing dependencies: {missing_deps}")
        for dep in missing_deps:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            except subprocess.CalledProcessError:
                print(f"Failed to install {dep}")
    else:
        print("All dependencies are already installed.")
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    print(f"GPU Available: {gpu_info['available']}")
    if gpu_info['available']:
        print(f"GPU Count: {gpu_info['count']}")
        print(f"GPU Name: {gpu_info['name']}")
    
    # Verify all dependencies again after installation
    final_deps_status = verify_dependencies()
    all_installed = all(final_deps_status.values())
    
    setup_summary = {
        'directories_created': True,
        'dependencies_installed': all_installed,
        'gpu_available': gpu_info['available'],
        'gpu_info': gpu_info
    }
    
    print("Environment setup complete!")
    return setup_summary


if __name__ == "__main__":
    setup_environment()