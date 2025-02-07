import os
import re
import requests
import boto3
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config

import boto3
from botocore.client import Config
from pathlib import Path
import pdb
import urllib.request

def download_icgc_object_storage(data_path, bucket_name="icgc25k-open", endpoint_url="https://object.genomeinformatics.org", files_to_download=None):
    """Download specific ICGC 25K Open data from object storage."""
    
    # Create S3 client with anonymous access
    s3 = boto3.client("s3", endpoint_url=endpoint_url, config=Config(signature_version=UNSIGNED))
    
    # List all files in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name)
    
    if "Contents" not in response:
        print("No files found in the bucket.")
        return
    

    # Ensure local directory exists
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # If no specific files are provided, download consensus_snv_indel
    #pdb.set_trace()

    if files_to_download is None:
        files_to_download = [obj["Key"] for obj in response["Contents"]]
    # Download specified files
    for file_key in files_to_download:
        local_file_path = data_path / file_key
        
        # Create subdirectories if needed
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading: {file_key}")
        s3.download_file(bucket_name, file_key, str(local_file_path))
    
    print("Download completed.")

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
        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"Successfully downloaded {genome_build} to {output_file}")
        except Exception as e:
            print(f"Error downloading {genome_build}: {str(e)}")

