import torch
import pdb
from pkg_resources import resource_filename
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from muat.util import *
import shutil
import os
import glob
import zipfile

def unziping_from_package_installation():
    pkg_ckpt = resource_filename('muat', 'pkg_ckpt')
    pkg_ckpt = ensure_dirpath(pkg_ckpt)

    all_zip = glob.glob(pkg_ckpt+'*.zip')
    if len(all_zip)>0:
        for checkpoint_file in all_zip:
            with zipfile.ZipFile(checkpoint_file, 'r') as zip_ref:
                zip_ref.extractall(path=pkg_ckpt)
            os.remove(checkpoint_file) 

def load_and_check_checkpoint(ckpt_path,save=False):

    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'),weights_only=False)

    if isinstance(checkpoint, dict):
        if 'target_handler' in checkpoint.keys():
            return checkpoint
        else:
            checkpoint = convert_checkpoint_version1(checkpoint,ckpt_path,save=save)
            return checkpoint
    else:
        if isinstance(checkpoint, list): #version 2 with args
            if len(checkpoint) == 3:
                checkpoint = convert_checkpoint_version2(checkpoint,ckpt_path,save=save)
                return checkpoint

def convert_checkpoint_version1(checkpoint,ckpt_path,save=False):
    print('convert checkpoint v.1')

    #args = get_checkpoint_args()

    ckpt_path = os.path.normpath(ckpt_path)
    splitfolder = ckpt_path.split('/')[-2].split('_')

    if 'pcawg' in ckpt_path:
        n_class = 24
    if 'tcga' in ckpt_path:
        n_class = 22

    smivn = splitfolder[1]

    if smivn == '10000':
        mut_type = 'SNV'
        mutratio = '1-0-0-0-0'
    elif smivn == '11000':
        mut_type = 'SNV+MNV'
        mutratio = '0.5-0.5-0-0-0'
    elif smivn == '11100':
        mut_type = 'SNV+MNV+indel'
        mutratio = '0.4-0.4-0.2-0-0'
    elif smivn == '11110':
        mut_type = 'SNV+MNV+indel+SV/MEI'
        mutratio='0.3-0.3-0.2-0.2-0'
    elif smivn == '11111':
        mut_type = 'SNV+MNV+indel+SV/MEI+Neg'
        mutratio='0.2-0.2-0.2-0.2-0.2'
    elif smivn == '10100':
        mut_type = 'SNV+indel'
        mutratio='0.5-0-0.5-0-0'
    #pdb.set_trace()
    mposges = splitfolder[2]
    
    if mposges == 'tripkon':
        motif=True
        motif_pos=False
        motif_pos_ges=False

        get_motif = True
        get_position = False
        get_ges = False

    if mposges == 'wpos':
        motif=False
        motif_pos=True
        motif_pos_ges=False

        get_motif = True
        get_position = True
        get_ges = False
        
    if mposges == 'wposges':
        motif=False
        motif_pos=False
        motif_pos_ges=True

        get_motif = True
        get_position = True
        get_ges = True        

    arch = splitfolder[3]
    
    if arch == 'CTransformerF':
        arch = 'MuAtMotifF'
    if arch == 'CTransformer':
        arch = 'MuAtMotifF'

    if arch == 'TripletPosition':
        arch = 'MuAtMotifPosition'
    if arch == 'TripletPositionF':
        arch = 'MuAtMotifPositionF'
    
    if arch == 'TripletPositionGES':
        arch = 'MuAtMotifPositionGES'
    if arch == 'TripletPositionGESF':
        arch = 'MuAtMotifPositionGESF'

    block_size = int(splitfolder[4][2:])

    n_layer = int(splitfolder[5][2:])

    n_head = int(splitfolder[6][2:])

    n_emb = int(splitfolder[7][2:])

    # Example usage:
    args = get_checkpoint_args()

    #fillin args
    args.mut_type = mut_type
    args.mutratio = mutratio
    args.motif = motif
    args.motif_pos =  motif_pos
    args.motif_pos_ges = motif_pos_ges
    args.arch = arch
    args.block_size = block_size
    args.n_layer = n_layer
    args.n_head = n_head
    args.n_emb = n_emb
    args.n_class = n_class
    args.get_motif = get_motif
    args.get_position = get_position
    args.get_ges = get_ges

    weight_newformat = [checkpoint,args,1]

    checkpoint = convert_checkpoint_version2(weight_newformat,ckpt_path,save)
    return checkpoint

def convert_checkpoint_version2(checkpoint,ckpt_path,save=False):
    print('convert checkpoint v.2')

    new_name = []

    if 'pcawg' in ckpt_path:
        new_name.append('pcawg')
        new_name.append('wgs')

    if 'tcga' in ckpt_path:
        new_name.append('tcga')
        new_name.append('wes')

    if '10000' in ckpt_path: 
        new_name.append('snv')
        mutation_type = 'snv'

    if '11000' in ckpt_path: 
        new_name.append('snv+mnv')
        mutation_type = 'snv+mnv'

    if '11100' in ckpt_path: 
        new_name.append('snv+mnv+indel')
        mutation_type = 'snv+mnv+indel'
    
    if '11110' in ckpt_path: 
        new_name.append('snv+mnv+indel+svmei')
        mutation_type = 'snv+mnv+indel+svmei'
    
    if '11111' in ckpt_path: 
        new_name.append('snv+mnv+indel+svmei+neg')
        mutation_type = 'snv+mnv+indel+svmei+neg'

    #check model
    model_name = ckpt_path.split('/')[-2].split('_')[3]

    if model_name == 'TripletPosition':
        model_name = 'MuAtMotifPosition'
        new_name.append(model_name)

    if model_name == 'TripletPositionF':
        model_name = 'MuAtMotifPositionF'
        new_name.append(model_name)

    if model_name == 'CTransformerF':
        model_name = 'MuAtMotifF'
        new_name.append(model_name)
    if model_name == 'CTransformer':
        model_name = 'MuAtMotif'
        new_name.append(model_name)

    if model_name == 'TripletPositionGES':
        model_name = 'MuAtMotifPositionGES'
        new_name.append(model_name)

    if model_name == 'TripletPositionGESF':
        model_name = 'MuAtMotifPositionGESF'
        new_name.append(model_name)

    #pdb.set_trace()

    if 'pcawg' in new_name:
        extdir = resource_filename('muat', 'extfile')
        extdir = ensure_dirpath(extdir)
        classfileinfo = extdir + 'classinfo_pcawg.tsv'
        le = LabelEncoderFromCSV(csv_file=classfileinfo,class_name_col='class_name',class_index_col='class_index')

    if 'tcga' in new_name:
        extdir = resource_filename('muat', 'extfile')
        extdir = ensure_dirpath(extdir)
        classfileinfo = extdir + 'classinfo_tcga.tsv'
        le = LabelEncoderFromCSV(csv_file=classfileinfo,class_name_col='class_name',class_index_col='class_index')

    #pdb.set_trace()
    dict_motif = pd.read_csv(extdir + 'dictMutation.tsv',sep='\t')
    dict_pos = pd.read_csv(extdir + 'dictChpos.tsv',sep='\t')
    dict_ges = pd.read_csv(extdir + 'dictGES.tsv',sep='\t')

    mutratio = checkpoint[1].mutratio.split('-')
    snv_ratio = float(mutratio[0])
    mnv_ratio = float(mutratio[1])
    indel_ratio = float(mutratio[2])
    sv_mei_ratio = float(mutratio[3])
    neg_ratio = float(mutratio[4])

    #pdb.set_trace()

    # Ensure these values are set correctly

    mutation_sampling_size = checkpoint[1].block_size
    n_emb = checkpoint[1].n_emb
    n_layer = checkpoint[1].n_layer
    n_head = checkpoint[1].n_head
    target_handler = []

    n_class = len(le.classes_)
    target_handler.append(le)

    model_config = ModelConfig(
                        model_name,
                        dict_motif,
                        dict_pos,
                        dict_ges,
                        mutation_sampling_size,
                        n_layer,
                        n_emb,
                        n_head,
                        n_class, 
                        mutation_type)
    
    trainer_config = TrainerConfig()

    weight = checkpoint[0]

    model = get_model(model_name,model_config)

    if checkpoint[1].motif:
        motif = True
        pos = False
        ges = False
    elif checkpoint[1].motif_pos:
        motif = True
        pos = True
        ges = False
    elif checkpoint[1].motif_pos_ges:
        motif = True
        pos = True
        ges = True

    #pdb.set_trace()
    dataloader_config = DataloaderConfig(model_input=model_config.model_input,mutation_type_ratio=model_config.mutation_type_ratio,mutation_sampling_size=mutation_sampling_size)

    save_ckpt_params = {'weight':weight,
                    'target_handler':target_handler,
                    'model_config':model_config,
                    'trainer_config':trainer_config,
                    'dataloader_config':dataloader_config,
                    'model_name':model.__class__.__name__,
                    'motif_dict':dict_motif,
                    'pos_dict':dict_pos,
                    'ges_dict':dict_ges}

    if save:
        old_dir = ensure_dirpath(os.path.dirname(ckpt_path))
        filename = '-'.join(new_name) + '.pthx'
        #pdb.set_trace()
        path = old_dir + filename
        path = os.path.normpath(path)
        torch.save(save_ckpt_params,path)
        print('the latest checkpoint version has been saved in ' + path + '. Previous checkpoint version is deprecated, use this new version instead!')
        return save_ckpt_params, path
    else:
        return save_ckpt_params