from setuptools import setup, find_packages
import os
import urllib.request
import zipfile
import pdb

# Function to download and extract the checkpoint
def download_checkpoint():
    checkpoint_url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/best_pcawg.zip"  # Replace with your checkpoint URL
    checkpoint_dir = os.path.join('muat', 'pkg_ckpt')  # Directory where checkpoint will be stored

    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_file = os.path.join(checkpoint_dir, 'my_checkpoint.zip')

    # Download the checkpoint if it doesn't already exist
    if not os.path.exists(checkpoint_file):
        print("Downloading checkpoint...")
        urllib.request.urlretrieve(checkpoint_url, checkpoint_file)

        # Optionally extract if it's a zip file
        with zipfile.ZipFile(checkpoint_file, 'r') as zip_ref:
            zip_ref.extractall(path=checkpoint_dir)
        print(f"Checkpoint downloaded and extracted to {checkpoint_dir}")

        os.remove(checkpoint_file)  
    else:
        print("Checkpoint already exists, skipping download.")

# Get list of shell scripts
shell_scripts = [os.path.join('muat/pkg_shell', f) for f in os.listdir('muat/pkg_shell') 
                if f.endswith('.sh')]

# Call download_checkpoint function to download the checkpoint during installation
download_checkpoint()

setup(
    name="muat",
    version="0.1.2",
    packages=find_packages(),
    package_data={
        'muat': [
            'pkg_data/*',
            'extfile/*',            
            'pkg_shell/*.sh',  # Make sure shell scripts are included as package data
            'pkg_ckpt/*'
        ],
    },
    scripts=shell_scripts,  # Install as executable scripts
    install_requires=[
        "numpy",
        "pandas",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "muat=muat.core:main",  # Change this to your actual CLI entry point
        ]
    },
    include_package_data=True,  # Important for conda packaging
)
