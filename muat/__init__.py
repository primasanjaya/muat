__version__ = '0.1.0'

# Import specific functions you want to expose
from .preprocessing import preprocessing_vcf
from .download import download_reference, download_icgc_object_storage
from .reader import (
    get_reader,
    process_input,
    Variant,
    VCFReader,
    MAFReader
)
from .util import (
    read_reference,
    read_codes,
    ensure_dir_exists,
    open_stream,
    # Add other utility functions you want to expose
)

# Import modules for hierarchical access
from . import download
from . import preprocessing
from . import reader
from . import util

# List all public functions/classes that should be available
__all__ = [
    # Preprocessing functions
    'preprocessing_vcf',
    
    # Download functions
    'download_reference',
    'download_icgc_object_storage',
    
    # Reader classes and functions
    'get_reader',
    'process_input',
    'Variant',
    'VCFReader',
    'MAFReader',
    
    # Utility functions
    'read_reference',
    'read_codes',
    'ensure_dir_exists',
    'open_stream',
    
    # Module imports
    'download',
    'preprocessing',
    'reader',
    'util'
]