import torch
import pdb
from pkg_resources import resource_filename
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from muat.util import *
import shutil

def check_checkpoint_and_fix(checkpoint,args):

    if isinstance(checkpoint, list):
        classfileinfo = resource_filename('muat', 'extfile')
        if args.classinfo_filepath is None or args.dict_motif_filepath is None or args.dict_pos_filepath is None or args.dict_ges_filepath is None:
            #raise ValueError("You are using old checkpoint version. Please provide --classinfo-filepath --dict-motif-filepath --dict-pos-filepath --dict-ges-filepath, example files are in ",classfileinfo)

            #converting old checkpoint
            print('pass')

        else:
            classfileinfo = args.classinfo_filepath
            target_handler = LabelEncoder()
            pd_classinfo = pd.read_csv(classfileinfo,index_col=0)
            target_handler.fit(pd_classinfo['class_name'])
            #pdb.set_trace()

            weight = checkpoint[0]

            dict_motif = pd.read_csv(args.dict_motif_filepath,sep='\t')
            dict_pos = pd.read_csv(args.dict_pos_filepath,sep='\t')
            dict_ges = pd.read_csv(args.dict_ges_filepath,sep='\t')

            #pdb.set_trace()

            mutratio = checkpoint[1].mutratio.split('-')
            snv_ratio = float(mutratio[0])
            mnv_ratio = float(mutratio[1])
            indel_ratio = float(mutratio[2])
            sv_mei_ratio = float(mutratio[3])
            neg_ratio = float(mutratio[4])

            # Ensure these values are set correctly
            mutation_type, motif_size = mutation_type_ratio(snv=snv_ratio, mnv=mnv_ratio, indel=indel_ratio, sv_mei=sv_mei_ratio, neg=neg_ratio, pd_motif=dict_motif)

            n_class = checkpoint[1].n_class
            mutation_sampling_size = checkpoint[1].block_size
            n_emb = checkpoint[1].n_emb
            n_layer = checkpoint[1].n_layer
            n_head = checkpoint[1].n_head

            # Check the values before creating the model
            model_config = ModelConfig(
                motif_size=motif_size+1,#plus one for padding
                num_class=n_class,
                mutation_sampling_size=mutation_sampling_size,
                position_size=len(dict_pos)+1,#plus one for padding 
                ges_size=len(dict_ges)+1,#plus one for padding
                n_embd=n_emb,
                n_layer=n_layer,
                n_head=n_head
            )
            
            trainer_config = TrainerConfig()

            model = get_model(checkpoint[1].arch,model_config)

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

            model_use = model_input(motif=motif,pos=pos,ges=ges) #model input

            dataloader_config = DataloaderConfig(model_input=model_use,mutation_type=mutation_type,mutation_sampling_size=mutation_sampling_size)

            save_ckpt_params = {'weight':weight,
                            'target_handler':target_handler,
                            'model_config':model_config,
                            'trainer_config':trainer_config,
                            'dataloader_config':dataloader_config,
                            'model':model,
                            'motif_dict':dict_motif,
                            'pos_dict':dict_pos,
                            'ges_dict':dict_ges}

            return save_ckpt_params
    else:
        return save_ckpt_params

def load_and_check_checkpoint(ckpt_path):

    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))

    #pdb.set_trace()

    if len(checkpoint) >=3: #version 1 no args
        checkpoint = convert_checkpoint_version1(checkpoint,ckpt_path)
    else:     
        if isinstance(checkpoint, list): #version 2 with args
            checkpoint = convert_checkpoint_version2(checkpoint,ckpt_path)
    
    return checkpoint

def convert_checkpoint_version1(checkpoint,ckpt_path):

    args = get_checkpoint_args()

    splitfolder = ckpt_path.split('/')[-2].split('_')

    if 'pcawg' in ckpt_path:
        n_class = 24
    if 'tcga' in ckpt_path:
        n_class = 22

    ##pdb.set_trace()

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

    convert_checkpoint_version2(weight_newformat,ckpt_path)

def convert_checkpoint_version2(checkpoint,ckpt_path):

    new_name = []

    if 'pcawg' in ckpt_path:
        new_name.append('pcawg')
        new_name.append('wgs')

    if 'tcga' in ckpt_path:
        new_name.append('tcga')
        new_name.append('wes')

    if '10000' in ckpt_path: 
        new_name.append('snv')

    if '11000' in ckpt_path: 
        new_name.append('snv+mnv')

    if '11100' in ckpt_path: 
        new_name.append('snv+mnv+indel')
    
    if '11110' in ckpt_path: 
        new_name.append('snv+mnv+indel+svmei')
    
    if '11111' in ckpt_path: 
        new_name.append('snv+mnv+indel+svmei+neg')

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
        classfileinfo = extdir + '/' + 'classinfo_pcawg.csv'
        target_handler = LabelEncoder()
        pd_classinfo = pd.read_csv(classfileinfo,index_col=0)
        target_handler.fit(pd_classinfo['class_name'])

    dict_motif = pd.read_csv(extdir + '/' + 'dictMutation.tsv',sep='\t')
    dict_pos = pd.read_csv(extdir + '/' + 'dictChpos.tsv',sep='\t')
    dict_ges = pd.read_csv(extdir + '/' + 'dictGES.tsv',sep='\t')

    mutratio = checkpoint[1].mutratio.split('-')
    snv_ratio = float(mutratio[0])
    mnv_ratio = float(mutratio[1])
    indel_ratio = float(mutratio[2])
    sv_mei_ratio = float(mutratio[3])
    neg_ratio = float(mutratio[4])

    # Ensure these values are set correctly
    mutation_type, motif_size = mutation_type_ratio(snv=snv_ratio, mnv=mnv_ratio, indel=indel_ratio, sv_mei=sv_mei_ratio, neg=neg_ratio, pd_motif=dict_motif)

    n_class = checkpoint[1].n_class
    mutation_sampling_size = checkpoint[1].block_size
    n_emb = checkpoint[1].n_emb
    n_layer = checkpoint[1].n_layer
    n_head = checkpoint[1].n_head

    # Check the values before creating the model
    model_config = ModelConfig(
        motif_size=motif_size+1,#plus one for padding
        num_class=n_class,
        mutation_sampling_size=mutation_sampling_size,
        position_size=len(dict_pos)+1,#plus one for padding 
        ges_size=len(dict_ges)+1,#plus one for padding
        n_embd=n_emb,
        n_layer=n_layer,
        n_head=n_head
    )
    
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

    model_use = model_input(motif=motif,pos=pos,ges=ges) #model input

    dataloader_config = DataloaderConfig(model_input=model_use,mutation_type=mutation_type,mutation_sampling_size=mutation_sampling_size)

    save_ckpt_params = {'weight':weight,
                    'target_handler':target_handler,
                    'model_config':model_config,
                    'trainer_config':trainer_config,
                    'dataloader_config':dataloader_config,
                    'model':model,
                    'motif_dict':dict_motif,
                    'pos_dict':dict_pos,
                    'ges_dict':dict_ges}

    #pdb.set_trace()
    new_name = '-'.join(new_name)

    torch.save(save_ckpt_params,'/csc/epitkane/projects/github/muat/muat/pkg_ckpt/' + new_name + '.pthx')

    