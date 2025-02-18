import sys
import os
import tarfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from muat.download import *
from muat.preprocessing import *
import glob
import pandas as pd
import pdb
from muat.util import *
from muat.dataloader import *
from muat.trainer import *
from muat.predict import *
from muat.model import *
from muat.checkpoint import *

if __name__ == "__main__":

    args = get_simlified_args()

    muat_dir = 'path/to/muat'
    
    genome_reference_38_path = 'path/to/genome_reference/hg38.fa.gz'
    genome_reference_19_path = 'path/to/genome_reference/hg19.fa.gz'

    #load ckpt
    load_ckpt_path = 'path/to/ckpt/weight.pthx'

    checkpoint = load_and_check_checkpoint(load_ckpt_path)

    dict_motif,dict_pos,dict_ges = load_token_dict(checkpoint)

    #example for preprocessing multiple vcf files
    vcf_files = 'path/to/input/sample.vcf.gz'
    tmp_dir = muat_dir + '/data/preprocessed_test/'
    vcf_files = multifiles_handler(vcf_files)
    preprocessing_vcf38_tokenizing(vcf_file=vcf_files,
                                genome_reference_38_path=genome_reference_38_path,
                                genome_reference_19_path=genome_reference_19_path,
                                tmp_dir=tmp_dir,
                                dict_motif=dict_motif,
                                dict_pos=dict_pos,
                                dict_ges=dict_ges)

    predict_ready_files = []
    for x in vcf_files:
        if os.path.exists(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')

    pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])

    le = load_target_handler(checkpoint)

    dataloader_config = checkpoint['dataloader_config']
    test_dataloader = MuAtDataloader( pd_predict,dataloader_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = checkpoint['model']
    model = model.to(device)
    model.load_state_dict(checkpoint['weight'])

    predict_config = PredictorConfig(max_epochs=1, batch_size=1,result_dir=os.path.dirname(load_ckpt_path),target_handler=le)
    #pdb.set_trace()
    predict_config.get_features = True #get features
    predictor = Predictor(model, test_dataloader, predict_config)

    predictor.batch_predict()

    print('result saved in ',os.path.dirname(load_ckpt_path))



