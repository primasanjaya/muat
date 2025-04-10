__version__ = '0.1.9'

# Import specific functions you want to expose
from .preprocessing import (
    preprocessing_vcf,
    preprocessing_vcf38,
    preprocessing_tsv38,
    preprocessing_vcf_tokenizing,
    preprocessing_vcf38_tokenizing,
    preprocessing_tsv38_tokenizing,
    combine_somagg_chunks_to_platekey,
    filtering_somagg_vcf,
    get_motif_pos_ges,
    load_dict,
    create_dictionary,
    tokenizing
)

from .download import (
    download_reference,
    download_icgc_object_storage,
    download_checkpoint
)

from .reader import (
    get_reader,
    process_input,
    align,
    get_context,
    Variant,
    VCFReader,
    MAFReader,
    SomAggTSVReader,
    VariantReader
)

from .util import (
    read_reference,
    read_codes,
    ensure_dir_exists,
    open_stream,
    get_sample_name,
    gunzip_file,
    status,
    is_valid_dna,
    get_checkpoint_args,
    mut_type_checkpoint_handler,
    check_model_match,
    initialize_pretrained_weight,
    get_model,
    multifiles_handler,
    load_token_dict,
    load_target_handler,
    openz,
    get_timestr
)

from .trainer import (
    Trainer,
    TrainerConfig
)

from .predict import (
    Predictor,
    PredictorConfig
)

from .checkpoint import (
    convert_checkpoint_v2tov3
)

# Import modules for hierarchical access
from . import download
from . import preprocessing
from . import reader
from . import util
from . import trainer
from . import predict
from . import checkpoint

# List all public functions/classes that should be available
__all__ = [
    # Preprocessing functions
    'preprocessing_vcf',
    'preprocessing_vcf38',
    'preprocessing_tsv38',
    'preprocessing_vcf_tokenizing',
    'preprocessing_vcf38_tokenizing',
    'preprocessing_tsv38_tokenizing',
    'combine_somagg_chunks_to_platekey',
    'filtering_somagg_vcf',
    'get_motif_pos_ges',
    'load_dict',
    'create_dictionary',
    'tokenizing',
    
    # Download functions
    'download_reference',
    'download_icgc_object_storage',
    'download_checkpoint',
    
    # Reader classes and functions
    'get_reader',
    'process_input',
    'align',
    'get_context',
    'Variant',
    'VCFReader',
    'MAFReader',
    'SomAggTSVReader',
    'VariantReader',
    
    # Utility functions
    'read_reference',
    'read_codes',
    'ensure_dir_exists',
    'open_stream',
    'get_sample_name',
    'gunzip_file',
    'status',
    'is_valid_dna',
    'get_checkpoint_args',
    'mut_type_checkpoint_handler',
    'check_model_match',
    'initialize_pretrained_weight',
    'get_model',
    'multifiles_handler',
    'load_token_dict',
    'load_target_handler',
    'openz',
    'get_timestr',
    
    # Trainer and Predictor classes
    'Trainer',
    'TrainerConfig',
    'Predictor',
    'PredictorConfig',
    
    # Checkpoint functions
    'convert_checkpoint_v2tov3',
    
    # Module imports
    'download',
    'preprocessing',
    'reader',
    'util',
    'trainer',
    'predict',
    'checkpoint'
]