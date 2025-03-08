"""Generate API documentation for libfuncs."""

import os
import sys
import subprocess

def generate_docs():
    """Generate Sphinx documentation for the libfuncs library."""
    # Ensure we're in the docs directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create Sphinx configuration if it doesn't exist
    if not os.path.exists('conf.py'):
        print("Initializing Sphinx documentation...")
        subprocess.run(["sphinx-quickstart", "--no-sep", "-p", "libfuncs", 
                       "-a", "libfuncs contributors", "-v", "0.1.0", "-l", "en"])
    
    # Create or update the API documentation
    print("Generating API documentation...")
    subprocess.run(["sphinx-apidoc", "-o", "api", "../libfuncs", "--force"])
    
    # Build the HTML documentation
    print("Building HTML documentation...")
    subprocess.run(["make", "html"])
    
    print("Documentation built successfully in _build/html/")

if __name__ == "__main__":
    generate_docs()
