from setuptools import setup, find_packages
import os
import urllib.request
import zipfile
import pdb
import gzip
import shutil
import sys

def download_reference(genome_reference_path="./data/genome_reference/"):
    """
    Download reference genome files from UCSC
    
    Args:
        genome_reference_path (str): Path to store reference files
    """
    # Create directory if it doesn't exist
    os.makedirs(genome_reference_path, exist_ok=True)
    
    # Define URLs for reference files
    references = {
        'hg19': 'https://ftp.sanger.ac.uk/pub/project/PanCancer/genome.fa.gz',
        'hg38': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
    }
    
    for genome_build, url in references.items():
        output_file = os.path.join(genome_reference_path, f"{genome_build}.fa.gz")
        
        # Skip if file already exists
        if os.path.exists(output_file):
            print(f"{genome_build} reference already exists at {output_file}")
            continue
            
        print(f"Downloading {genome_build} reference...")
        
        urllib.request.urlretrieve(url, output_file)
        print(f"Successfully downloaded {genome_build} to {output_file}")
        
        try:
            # Gunzip the downloaded file
            with gzip.open(output_file, 'rb') as f_in:
                with open(os.path.join(genome_reference_path, f"{genome_build}.fa"), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Successfully gunzipped {genome_build} to {genome_reference_path}{genome_build}.fa")
            os.remove(output_file)
            print(f"Removed original gzipped file: {output_file}")
        except:
            os.remove(output_file)
            print(f"Removed original gzipped file: {output_file}")
            pass
            

# Function to download and extract the checkpoint
def download_checkpoint(url,name,extract=False):
    checkpoint_url = url  # Replace with your checkpoint URL
    checkpoint_dir = os.path.join('muat', 'pkg_ckpt')  # Directory where checkpoint will be stored

    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_file = os.path.join(checkpoint_dir, name)

    # Download the checkpoint if it doesn't already exist
    print("Downloading checkpoint...")
    urllib.request.urlretrieve(checkpoint_url, checkpoint_file)

    # Optionally extract if it's a zip file
    if extract:
        with zipfile.ZipFile(checkpoint_file, 'r') as zip_ref:
            zip_ref.extractall(path=checkpoint_dir)
        print(f"Checkpoint downloaded and extracted to {checkpoint_dir}")
        os.remove(checkpoint_file)  

# Get list of shell scripts
shell_scripts = [os.path.join('muat/pkg_shell', f) for f in os.listdir('muat/pkg_shell') 
                if f.endswith('.sh')]

download_checkpoint("https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/best_wgs_pcawg.zip",'best_wgs_pcawg.zip',False)
download_checkpoint("https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/benchmark_wes_slim.zip",'benchmark_wes_slim.zip',False)

#download genome reference and unzip 
path = os.path.dirname(os.path.abspath(__file__)) + '/'
#download_reference(genome_reference_path=path + 'muat/genome_reference/')

setup(
    name="muat",
    version="0.1.11",
    packages=find_packages(),
    package_data={
        'muat': [
            'pkg_data/*',
            'extfile/*',            
            'pkg_shell/*.sh',  # Make sure shell scripts are included as package data
            'pkg_ckpt/*',
            'genome_reference/*'
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
