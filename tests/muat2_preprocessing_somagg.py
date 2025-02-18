from muat.util import *
from muat.preprocessing import *
import glob

#for faster preprocessing, submit the job to HPC per chunk file
tmp_dir = 'path/to/preprocessed_directory/'

somagg_chunks = '/path/to/somagg/chunks/somAgg_dr12_chr15_24318775_26757263.vcf.gz'

#filtering somagg with pass and . only
filtering_somagg_vcf(all_somagg_chunks,tmp_dir)

chuck_file = tmp_dir + 'somAgg_dr12_chr15_24318775_26757263.tsv'
split_chunk_persample(chuck_file,tmp_dir) #split chunk file to samples (saved in folder for paralel job processing)

sample_folder = glob.glob(tmp_dir + '/*')
combine_samplefolders_tosingle_tsv(sample_folder,tmp_dir) #combinin all sample folder to singe tsv file conatining all mutations  per sample

tsv_file = tmp_dir + 'somAgg_dr12_chr15_24318775_26757263.somagg.tsv.gz'
muat_dir = '/path/to/muat/'

genome_reference_38_path = '/path/to/genome_reference/hg38.fa.gz'
genome_reference_19_path = '/path/to/genome_reference/hg19.fa.gz'

#get motif pos and ges annotation from .tsv somagg and tokenize
dict_motif = pd.read_csv(muat_dir + '/muat/extfile/dictMutation.tsv',sep='\t')
dict_pos = pd.read_csv(muat_dir + '/muat/extfile/dictChpos.tsv',sep='\t')
dict_ges = pd.read_csv(muat_dir + '/muat/extfile/dictGES.tsv',sep='\t')
preprocessing_tsv38_tokenizing(tsv_file,genome_reference_38_path,genome_reference_19_path,tmp_dir,dict_motif,dict_pos,dict_ges)