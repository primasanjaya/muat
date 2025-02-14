from muat.util import *
from muat.preprocessing import *
import glob

all_somagg_chunks = '/gel_data_resources/main_programme/aggregation/aggregated_somatic_strelka/somAgg/v0.2/genomic_data/*.vcf.gz'

all_somagg_chunks = '/Users/primasan/Downloads/CRC_testsample_GRCh38_Mutect2_PASS_MSS.vcf.gz'
all_somagg_chunks = multifiles_handler(all_somagg_chunks)

tmp_dir = '/path/to/tmp_dir/'

filtering_somagg_vcf(all_somagg_chunks,tmp_dir)

all_filtered_chunks = glob.glob(tmp_dir + '*.tsv')
combine_chunk_persample(all_filtered_chunks,tmp_dir)

