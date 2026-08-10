import torch
import pdb
from muat._resources import pkg_path
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from muat.util import *
import shutil
import os
import glob
import zipfile
import json
from muat.download import download_checkpoint

import tempfile

from muat.model import ModelConfig
from muat.trainer import TrainerConfig
from muat.dataloader import DataloaderConfig
from muat.util import LabelEncoderFromCSV

def unziping_from_package_installation():
    pkg_ckpt = pkg_path('pkg_ckpt')
    pkg_ckpt = ensure_dirpath(pkg_ckpt)

    all_zip = glob.glob(pkg_ckpt+'*.zip')
    if len(all_zip)>0:
        for checkpoint_file in all_zip:
            with zipfile.ZipFile(checkpoint_file, 'r') as zip_ref:
                zip_ref.extractall(path=pkg_ckpt)
            os.remove(checkpoint_file) 

def make_json_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")  # List of row dicts
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

def save_model_config_to_json(config, filepath: str):
    serialisable_dict = {
        k: make_json_serializable(v)
        for k, v in config.__dict__.items()
    }
    with open(filepath, "w") as f:
        json.dump(serialisable_dict,f)

def save_dict_to_json(data, filepath: str):
    """Helper function to save dictionary data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)

def save_dataframe_to_json(df, filepath: str):
    """Helper function to save pandas DataFrame to JSON file."""
    data = df.to_dict(orient="records")
    save_dict_to_json(data, filepath)

def convert_checkpoint_v2tov3(ckpt_path, save_dir,checkpoint = None, save = True):
    print('convert checkpoint v.3')
    # Load the old checkpoint
    if checkpoint is None:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save weights
    weights_path = os.path.join(save_dir, 'weight.pth')
    torch.save(checkpoint['weight'], weights_path)

    # Save target handlers
    for idx, handler in enumerate(checkpoint['target_handler']):
        filepath = os.path.join(save_dir, f'target_handler_{idx+1}.json')
        save_dict_to_json({
            "class_to_idx": handler.class_to_idx,
            "idx_to_class": handler.idx_to_class,
            "classes_": handler.classes_
        }, filepath)

    # Save configs
    configs = {
        'model_config': checkpoint['model_config'],
        'trainer_config': checkpoint['trainer_config'],
        'dataloader_config': checkpoint['dataloader_config']
    }
    
    for name, config in configs.items():
        filepath = os.path.join(save_dir, f'{name}.json')
        save_model_config_to_json(config, filepath)

    # Save model name
    save_dict_to_json(checkpoint['model_name'], os.path.join(save_dir, 'model_name.json'))

    # Save dictionaries
    dicts = {
        'motif_dict': checkpoint['motif_dict'],
        'pos_dict': checkpoint['pos_dict'],
        'ges_dict': checkpoint['ges_dict']
    }
    
    for name, df in dicts.items():
        filepath = os.path.join(save_dir, f'{name}.json')
        save_dataframe_to_json(df, filepath)

    # Reload all JSON files into a new checkpoint dictionary
    new_checkpoint = {}
    
    # Load weights
    new_checkpoint['weight'] = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)

    new_checkpoint['target_handler'] = []
    target_handler_files = [f for f in os.listdir(save_dir) if f.startswith('target_handler_')]
    for file in sorted(target_handler_files):
        filepath = os.path.join(save_dir, file)
        handler = LabelEncoderFromCSV.from_json(filepath)
        new_checkpoint['target_handler'].append(handler)
    
    # Load configs and reconstruct classes
    with open(os.path.join(save_dir, 'model_config.json'), 'r') as f:
        model_config_data = json.load(f)
        # Convert DataFrames back from records
        model_config_data['dict_motif'] = pd.DataFrame(model_config_data['dict_motif'])
        model_config_data['dict_pos'] = pd.DataFrame(model_config_data['dict_pos'])
        model_config_data['dict_ges'] = pd.DataFrame(model_config_data['dict_ges'])
        model_config_data['n_class'] = model_config_data['num_class']
        model_config_data['n_emb'] = model_config_data['n_embd']
        new_checkpoint['model_config'] = ModelConfig(**model_config_data)
    
    with open(os.path.join(save_dir, 'trainer_config.json'), 'r') as f:
        trainer_config_data = json.load(f)
        new_checkpoint['trainer_config'] = TrainerConfig(**trainer_config_data)
    
    with open(os.path.join(save_dir, 'dataloader_config.json'), 'r') as f:
        dataloader_config_data = json.load(f)
        new_checkpoint['dataloader_config'] = DataloaderConfig(**dataloader_config_data)
    
    # Load model name
    with open(os.path.join(save_dir, 'model_name.json'), 'r') as f:
        new_checkpoint['model_name'] = json.load(f)
    
    # Load dictionaries
    for name in ['motif_dict', 'pos_dict', 'ges_dict']:
        with open(os.path.join(save_dir, f'{name}.json'), 'r') as f:
            new_checkpoint[name] = pd.DataFrame(json.load(f))

    if save:
        # Compare checkpoints
        print("\nComparing checkpoints:")
        for key in checkpoint.keys():
            ck_val = checkpoint[key]
            new_ck_val = new_checkpoint[key]
            
            if key == 'weight':
                # For OrderedDict of tensors, compare each tensor
                if isinstance(ck_val, dict) and isinstance(new_ck_val, dict):
                    is_equal = True
                    for k in ck_val.keys():
                        if k not in new_ck_val:
                            is_equal = False
                            break
                        if not torch.equal(ck_val[k], new_ck_val[k]):
                            is_equal = False
                            break
                else:
                    is_equal = False
                print(f"{key}: {'✓' if is_equal else '✗'}")
            elif key in ['motif_dict', 'pos_dict', 'ges_dict']:
                # For DataFrames, compare values
                is_equal = ck_val.equals(new_ck_val)
                print(f"{key}: {'✓' if is_equal else '✗'}")
            elif key == 'target_handler':
                # For target handlers, compare their attributes
                is_equal = all(
                    h1.class_to_idx == h2.class_to_idx and
                    h1.idx_to_class == h2.idx_to_class and
                    h1.classes_ == h2.classes_
                    for h1, h2 in zip(ck_val, new_ck_val)
                )
                print(f"{key}: {'✓' if is_equal else '✗'}")
            else:
                try:
                    all_ck_keys = ck_val.__dict__.keys()
                    if len(all_ck_keys) > 0:
                        for x in all_ck_keys:
                            cval = vars(ck_val)[x]
                            cnval = vars(new_ck_val)[x]

                            if isinstance(cval, pd.DataFrame):
                                # For DataFrames, use equals() method
                                is_equal = cval.equals(cnval)
                            else:
                                # For other types, use direct comparison
                                is_equal = cval == cnval
                            
                        print(f"{key}: {'✓' if is_equal else '✗'}")
                except:
                    is_equal = True
                    print(f"{key}: {'✓' if is_equal else '✗'}")

        #pdb.set_trace()
        # Create zip file
        zipfile = ckpt_path.split('/')[-1].split('.')[0] + 'v3.pthx'
        zip_checkpoint_files(save_dir,zipfile)
    
    return new_checkpoint

def load_checkpoint_v3(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    if not zipfile.is_zipfile(ckpt_path):
        raise ValueError(f"Checkpoint is not a valid .pthx zip file: {ckpt_path}")

    tmp_dir = tempfile.mkdtemp(prefix="muat_ckpt_")

    try:
        with zipfile.ZipFile(ckpt_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        required_files = [
            "weight.pth",
            "model_config.json",
            "trainer_config.json",
            "dataloader_config.json",
            "model_name.json",
            "motif_dict.json",
            "pos_dict.json",
            "ges_dict.json",
        ]

        for fname in required_files:
            fpath = os.path.join(tmp_dir, fname)
            if not os.path.exists(fpath):
                raise ValueError(f"Checkpoint is incomplete. Missing file: {fname}")

        new_checkpoint = {}

        weights_path = os.path.join(tmp_dir, "weight.pth")
        new_checkpoint["weight"] = torch.load(
            weights_path,
            map_location=torch.device("cpu"),
            weights_only=False
        )

        new_checkpoint["target_handler"] = []
        target_handler_files = sorted(
            f for f in os.listdir(tmp_dir) if f.startswith("target_handler_")
        )

        if len(target_handler_files) == 0:
            raise ValueError("Checkpoint does not contain any target_handler_*.json files.")

        for file in target_handler_files:
            filepath = os.path.join(tmp_dir, file)
            handler = LabelEncoderFromCSV.from_json(filepath)
            new_checkpoint["target_handler"].append(handler)

        with open(os.path.join(tmp_dir, "motif_dict.json"), "r") as f:
            motif_dict = pd.DataFrame(json.load(f))
        with open(os.path.join(tmp_dir, "pos_dict.json"), "r") as f:
            pos_dict = pd.DataFrame(json.load(f))
        with open(os.path.join(tmp_dir, "ges_dict.json"), "r") as f:
            ges_dict = pd.DataFrame(json.load(f))

        new_checkpoint["motif_dict"] = motif_dict
        new_checkpoint["pos_dict"] = pos_dict
        new_checkpoint["ges_dict"] = ges_dict

        with open(os.path.join(tmp_dir, "model_config.json"), "r") as f:
            model_config_data = json.load(f)

        model_config_data["dict_motif"] = motif_dict
        model_config_data["dict_pos"] = pos_dict
        model_config_data["dict_ges"] = ges_dict

        new_checkpoint["model_config"] = ModelConfig(**model_config_data)

        with open(os.path.join(tmp_dir, "trainer_config.json"), "r") as f:
            trainer_config_data = json.load(f)
        new_checkpoint["trainer_config"] = TrainerConfig(**trainer_config_data)

        with open(os.path.join(tmp_dir, "dataloader_config.json"), "r") as f:
            dataloader_config_data = json.load(f)
        new_checkpoint["dataloader_config"] = DataloaderConfig(**dataloader_config_data)

        with open(os.path.join(tmp_dir, "model_name.json"), "r") as f:
            new_checkpoint["model_name"] = json.load(f)

        metadata_path = os.path.join(tmp_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                new_checkpoint["metadata"] = json.load(f)

        return new_checkpoint

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def _backfill_config_defaults(checkpoint):
    """Backfill config fields added after an older checkpoint was pickled.

    Pickle restores instances without calling __init__, so a ModelConfig /
    DataloaderConfig saved by an older muat version is missing any attribute
    introduced later. We re-add them with their current defaults so the model
    can be reconstructed. The dropout fields are inert at predict time
    (model.eval()); n_emb/n_class are aliases of n_embd/num_class."""
    mc = checkpoint.get('model_config')
    if mc is not None:
        for attr, default in (('embd_pdrop', 0.1), ('resid_pdrop', 0.1), ('attn_pdrop', 0.1)):
            if not hasattr(mc, attr):
                setattr(mc, attr, default)
        if not hasattr(mc, 'n_emb') and hasattr(mc, 'n_embd'):
            mc.n_emb = mc.n_embd
        if not hasattr(mc, 'n_class') and hasattr(mc, 'num_class'):
            mc.n_class = mc.num_class
    dc = checkpoint.get('dataloader_config')
    if dc is not None and not hasattr(dc, 'sampling_replacement'):
        dc.sampling_replacement = False
    return checkpoint


def load_and_check_checkpoint(ckpt_path, save=False):
    if not ckpt_path:
        raise ValueError("No checkpoint path was provided.")

    ckpt_path = resolve_path(ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    try:
        checkpoint = torch.load(
            ckpt_path,
            map_location=torch.device("cpu"),
            weights_only=False
        )

        if isinstance(checkpoint, dict):
            if "target_handler" in checkpoint:
                return _backfill_config_defaults(checkpoint)
            checkpoint = convert_checkpoint_version1(checkpoint, ckpt_path, save=save)
            return checkpoint

        if isinstance(checkpoint, list) and len(checkpoint) == 3:
            checkpoint = convert_checkpoint_version2(checkpoint, ckpt_path, save=save)
            return checkpoint

        raise ValueError(f"Unsupported legacy checkpoint structure in: {ckpt_path}")

    except Exception as legacy_error:
        try:
            checkpoint = load_checkpoint_v3(ckpt_path)
            return checkpoint
        except Exception as v3_error:
            raise ValueError(
                f"Failed to load checkpoint: {ckpt_path}\n"
                f"Legacy loader error: {legacy_error}\n"
                f"V3 loader error: {v3_error}\n"
                "The checkpoint may be corrupted, incomplete, or incompatible."
            )

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
        extdir = pkg_path('extfile')
        extdir = ensure_dirpath(extdir)
        classfileinfo = extdir + 'classinfo_pcawg.tsv'
        le = LabelEncoderFromCSV(csv_file=classfileinfo,class_name_col='class_name',class_index_col='class_index')

    if 'tcga' in new_name:
        extdir = pkg_path('extfile')
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
    
    old_dir = ensure_dirpath(os.path.dirname(ckpt_path))

    if save:
        filename = '-'.join(new_name) + '.pthx'
        #pdb.set_trace()
        path = old_dir + filename
        path = os.path.normpath(path)
        torch.save(save_ckpt_params,path)
        print('the latest checkpoint version has been saved in ' + path + '. Previous checkpoint version is deprecated, use this new version instead!')
        return save_ckpt_params, path

    checkpoint_v3 = convert_checkpoint_v2tov3(ckpt_path,old_dir,save_ckpt_params,True)
    return checkpoint_v3

def zip_checkpoint_files(directory: str, zip_name: str) -> str:
    """
    Zip all .json and .pth files in the specified directory.
    
    Args:
        directory (str): Path to the directory containing files to zip
        zip_name (str): Name for the zip file. If None, will use directory name.
        
    Returns:
        str: Path to the created zip file
    """
    directory = ensure_dirpath(directory)
    
    # Get all .json and .pth files
    files_to_zip = []
    for ext in ['.json', '.pth']:
        files_to_zip.extend(glob.glob(os.path.join(directory, f'*{ext}')))
    
    if not files_to_zip:
        raise ValueError(f"No .json or .pth files found in {directory}")
        
    zip_path = os.path.join(directory, zip_name)    
    # Create zip file
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file, os.path.basename(file))

    #pdb.set_trace()
    for x in files_to_zip:
        os.remove(x)
    
    return zip_path

def unzip_checkpoint_files(zip_path: str, extract_dir: str) -> str:
    """
    Unzip a checkpoint file to the specified directory.
    
    Args:
        zip_path (str): Path to the zip file
        extract_dir (str, optional): Directory to extract files to. If None, will use the same directory as the zip file.
        
    Returns:
        str: Path to the directory where files were extracted
    """
    if not os.path.exists(zip_path):
        raise ValueError(f"Zip file not found: {zip_path}")
    
    extract_dir = ensure_dirpath(extract_dir)
    
    # Create extract directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract all files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path=extract_dir)
    
    return extract_dir