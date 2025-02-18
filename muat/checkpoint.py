import torch
import pdb
from pkg_resources import resource_filename
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from muat.util import *

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

    checkpoint = torch.load(ckpt_path)
    
    if isinstance(checkpoint, list):
        checkpoint = convert_checkpoint(checkpoint,ckpt_path)
    
    return checkpoint

def convert_checkpoint(checkpoint,ckpt_path):

    new_name = []

    if 'pcawg' in ckpt_path:
        new_name.append('pcawg')

    if 'wgs' in ckpt_path:
        new_name.append('wgs')

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

    if model_name == 'TripletPositionF':
        model_name = 'MuAtMotifPositionF'
        new_name.append(model_name)

    new_name = '-'.join(new_name)

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

    torch.save(save_ckpt_params, os.path.dirname(ckpt_path) + '/' + new_name + '.pthx')
    