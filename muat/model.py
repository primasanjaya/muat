import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

logger = logging.getLogger(__name__)

class ModelConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self,model_name,
                    dict_motif,
                    dict_pos,
                    dict_ges,
                    mutation_sampling_size,
                    n_layer,
                    n_emb,
                    n_head,
                    n_class,
                    mutation_type, 
                    **kwargs):

        self.model_name = model_name
        self.dict_motif = dict_motif
        self.dict_pos = dict_pos
        self.dict_ges = dict_ges
        self.mutation_sampling_size = mutation_sampling_size
        self.n_layer = n_layer
        self.n_embd = n_emb
        self.n_head = n_head
        self.num_class = n_class
        self.mutation_type = mutation_type

        self.model_input = self.input_handler(model_name)
        self.position_size = len(self.dict_pos)+1 #plus one for padding
        self.ges_size = len(self.dict_ges)+1 #plus one for padding

        self.mutation_type_ratio = self.get_mut_ratio(mutation_type)
        motif_size = self.compute_motif_size(dict_motif,self.mutation_type_ratio)
        
        self.motif_size = motif_size + 1  # plus one for padding
        #pdb.set_trace()
   
        for k,v in kwargs.items():
            setattr(self, k, v)

    def input_handler(self,arch):
        motif = False
        pos = False
        ges = False
        if 'motif' in arch.lower():
            motif = True
        if 'position' in arch.lower():
            pos = True
        if 'ges' in arch.lower():
            ges = True
        return {'motif': motif,
                'pos': pos,
                'ges': ges
                }

    def compute_motif_size(self,pd_motif,mutatation_type_ratio):

        vocabsize = 0
        vocabSNV = len(pd_motif.loc[pd_motif['mut_type']=='SNV'])
        vocabMNV = len(pd_motif.loc[pd_motif['mut_type']=='MNV'])
        vocabindel = len(pd_motif.loc[pd_motif['mut_type']=='indel']) 
        vocabSVMEI = len(pd_motif.loc[pd_motif['mut_type'].isin(['MEI','SV'])])
        vocabNormal = len(pd_motif.loc[pd_motif['mut_type']=='Normal'])

        snv,mnv,indel,sv_mei,neg = mutatation_type_ratio.values()
        if snv>0:
            vocabsize = vocabSNV
        if mnv>0:
            vocabsize = vocabSNV + vocabMNV
        if indel>0:
            vocabsize = vocabSNV + vocabMNV + vocabindel         
        if sv_mei>0:
            vocabsize = vocabSNV + vocabMNV + vocabindel + vocabSVMEI   
        if neg>0:
            vocabsize = vocabSNV + vocabMNV + vocabindel + vocabSVMEI + vocabNormal

        return vocabsize
        

    def get_mut_ratio(self,mutation_type):
        if mutation_type == 'snv':
            mutation_type_ratio = {'snv': 1,'mnv': 0,'indel': 0,'sv_mei': 0,'neg': 0}
        elif mutation_type == 'snv+mnv':
            mutation_type_ratio =  {'snv': 0.5,'mnv': 0.5,'indel': 0,'sv_mei': 0,'neg': 0}
        elif mutation_type == 'snv+mnv+indel':
            mutation_type_ratio = {'snv': 0.4,'mnv': 0.4,'indel': 0.2,'sv_mei': 0,'neg': 0}
        elif mutation_type == 'snv+mnv+indel+svmei':
            mutation_type_ratio = {'snv': 0.3,'mnv': 0.4,'indel': 0.2,'sv_mei': 0.2,'neg': 0}
        elif mutation_type == 'snv+mnv+indel+svmei+neg':
            mutation_type_ratio={'snv': 0.2,'mnv': 0.2,'indel': 0.2,'sv_mei': 0.2,'neg': 0.2}
        elif mutation_type == 'snv+indel':
            mutation_type_ratio={'snv': 0.5,'mnv': 0,'indel': 0.5,'sv_mei': 0,'neg': 0}

        return mutation_type_ratio

        

class MuAtMotif(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.motif_size, False

        self.token_embedding = nn.Embedding(config.motif_size, config.n_embd, padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd, heads=config.n_head, seq_length=config.mutation_sampling_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(config.n_embd, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x, targets=None, vis=None,get_features=False):

        motif = x[:, 0, :]

        tokens = self.token_embedding(motif)

        b, t, e = tokens.size()

        x = self.do(tokens)

        x = self.tblocks(x)

        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over the time dimension

        logits = self.toprobs(x)

        if get_features:
            return Error('get_features is not implemented for MuAtMotif')

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

class MuAtMotifF(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.motif_size, False

        self.token_embedding = nn.Embedding(config.motif_size, config.n_embd, padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd, heads=config.n_head, seq_length=config.mutation_sampling_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd), 24),
                                       nn.ReLU())

        self.toprobs = nn.Linear(24, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x, targets=None, vis=None,get_features=False):

        motif = x[:, 0, :]

        tokens = self.token_embedding(motif)

        b, t, e = tokens.size()

        x = self.do(tokens)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over the time dimension

        feature = self.tofeature(x)

        if get_features:
            return feature
        else:
            logits = self.toprobs(feature)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class MuAtMotifF_2Labels(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.motif_size, False

        self.token_embedding = nn.Embedding(config.motif_size, config.n_embd, padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd, heads=config.n_head, seq_length=config.mutation_sampling_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)
        self.do = nn.Dropout(config.embd_pdrop)

        self.to_joinfeatures = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd + 4), 64),
                                       nn.ReLU())

        self.to_typefeatures = nn.Sequential(nn.Linear(64, 32),
                                       nn.ReLU())

        self.to_subtypefeatures = nn.Sequential(nn.Linear(64, 32),
                                       nn.ReLU())

        self.to_typeprobs = nn.Linear(32, config.num_class)

        self.to_subtypeprobs = nn.Linear(32, config.num_subclass)        

    def forward(self, x, targets=None, vis=None,get_features=False):

        motif = x[:, 0, :]

        tokens = self.token_embedding(motif)

        b, t, e = tokens.size()

        x = self.do(tokens)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over the time dimension

        join_features = self.to_joinfeatures(x)
        type_features = self.to_typefeatures(join_features)
        subtype_features = self.to_subtypefeatures(join_features)

        typeprobs = self.to_typeprobs(type_features)
        subtypeprobs = self.to_subtypeprobs(subtype_features)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss1 = F.cross_entropy(typeprobs.view(-1, typeprobs.size(-1)), targets.view(-1))
            loss2 = F.cross_entropy(subtypeprobs.view(-1, subtypeprobs.size(-1)), targets.view(-1))
            loss = loss1 + loss2

        logits_feats = {'first_logits': typeprobs,
                        'second_logits': subtypeprobs,
                        'first_features': type_features,
                        'second_features': subtype_features,
                        'join_features': join_features
                        }
        return logits_feats, loss

class MuAtMotifPosition(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.motif_size, False

        self.token_embedding = nn.Embedding(config.motif_size, config.n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd, padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd), heads=config.n_head, seq_length=config.mutation_sampling_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(int(config.n_embd + config.n_embd), config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x, targets=None, vis=None,get_features=False):

        triplettoken = x[:, 0, :]
        # pdb.set_trace()
        postoken = x[:, 1, :]

        tokens = self.token_embedding(triplettoken)
        positions = self.position_embedding(postoken)

        x = torch.cat((tokens, positions), axis=2)

        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over the time dimension

        logits = self.toprobs(x)

        if get_features:
            return Error('get_features is not implemented for MuAtMotifPosition')

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class MuAtMotifPositionF(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.config = config 

        self.num_tokens, self.max_pool = config.motif_size, False
        self.token_embedding = nn.Embedding(config.motif_size, config.n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd, padding_idx=0)
        # pdb.set_trace()

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd), heads=config.n_head, seq_length=config.mutation_sampling_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd), 24),
                                       nn.ReLU())

        # pdb.set_trace()
        self.toprobs = nn.Linear(24, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x, targets=None, vis=None, visatt=None):

        triplettoken = x[:, 0, :]
        # pdb.set_trace()
        postoken = x[:, 1, :]

        tokens = self.token_embedding(triplettoken)
        positions = self.position_embedding(postoken)

        x = torch.cat((tokens, positions), axis=2)

        x = self.do(x)

        if visatt:
            dot1 = self.tblocks[0].attention(x, vis=True)

            manual_tblock = self.tblocks[0](x)

            after_tblock = self.tblocks(x)

            '''
            for block in self.tblocks:

                dot = block.attention(x,vis=True)

                pdb.set_trace()

                print(name)
            '''

            return dot1

        else:
            x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over the time dimension

        feature = self.tofeature(x)
        logits = self.toprobs(feature)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        logits_feats = {'first_logits': logits,
                        'first_features': feature
                        }
                        
        return logits_feats, loss

class MuAtMotifPositionF_2Labels(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.motif_size, False

        self.token_embedding = nn.Embedding(config.motif_size, config.n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd, padding_idx=0)

        # pdb.set_trace()

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd), heads=config.n_head, seq_length=config.mutation_sampling_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        self.to_joinfeatures = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd), 64),
                                       nn.ReLU())

        self.to_typefeatures = nn.Sequential(nn.Linear(64, 32),
                                       nn.ReLU())

        self.to_subtypefeatures = nn.Sequential(nn.Linear(64, 32),
                                       nn.ReLU())

        self.to_typeprobs = nn.Linear(32, config.num_class)

        self.to_subtypeprobs = nn.Linear(32, config.num_subclass)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x, targets=None, vis=None, visatt=None,get_features=False):

        triplettoken = x[:, 0, :]
        # pdb.set_trace()
        postoken = x[:, 1, :]

        tokens = self.token_embedding(triplettoken)
        positions = self.position_embedding(postoken)

        x = torch.cat((tokens, positions), axis=2)

        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) 

        join_features = self.to_joinfeatures(x)
        type_features = self.to_typefeatures(join_features)
        subtype_features = self.to_subtypefeatures(join_features)

        typeprobs = self.to_typeprobs(type_features)
        subtypeprobs = self.to_subtypeprobs(subtype_features)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            #pdb.set_trace()
            loss1 = F.cross_entropy(typeprobs.view(-1, typeprobs.size(-1)), targets[0].view(-1))
            loss2 = F.cross_entropy(subtypeprobs.view(-1, subtypeprobs.size(-1)), targets[1].view(-1))
            loss = loss1 + loss2

        logits_feats = {'first_logits': typeprobs,
                        'second_logits': subtypeprobs,
                        'first_features': type_features,
                        'second_features': subtype_features,
                        'join_features': join_features
                        }
        return logits_feats, loss

class MuAtMotifPositionGES(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.motif_size, False

        self.token_embedding = nn.Embedding(config.motif_size, config.n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd, padding_idx=0)
        self.ges_embedding = nn.Embedding(config.ges_size + 1, 4, padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd + 4), heads=config.n_head, seq_length=config.mutation_sampling_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(int(config.n_embd + config.n_embd + 4), config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x, targets=None, vis=None):

        triplettoken = x[:, 0, :]
        postoken = x[:, 1, :]
        gestoken = x[:, 2, :]

        tokens = self.token_embedding(triplettoken)
        positions = self.position_embedding(postoken)
        ges = self.ges_embedding(gestoken)

        x = torch.cat((tokens, positions, ges), axis=2)

        x = self.do(x)
        # pdb.set_trace()

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over the time dimension

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        logits_feats = {'first_logits': logits,
                        }
        return logits_feats, loss

class MuAtMotifPositionGESF(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.motif_size, False

        self.token_embedding = nn.Embedding(config.motif_size, config.n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd, padding_idx=0)
        self.ges_embedding = nn.Embedding(config.ges_size + 1, 4, padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd + 4), heads=config.n_head, seq_length=config.mutation_sampling_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd + 4), 24),
                                       nn.ReLU())

        self.toprobs = nn.Linear(24, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x, targets=None, vis=None,get_features=False):

        triplettoken = x[:, 0, :]
        postoken = x[:, 1, :]
        gestoken = x[:, 2, :]

        tokens = self.token_embedding(triplettoken)
        positions = self.position_embedding(postoken)
        ges = self.ges_embedding(gestoken)

        x = torch.cat((tokens, positions, ges), axis=2)

        x = self.do(x)
        # pdb.set_trace()

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over the time dimension

        feature = self.tofeature(x)
        logits = self.toprobs(feature)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        logits_feats = {'first_logits': logits,
                        'first_features': feature
                        }
                        
        return logits_feats, loss

class MuAtMotifPositionGESF_2Labels(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.motif_size, False

        self.token_embedding = nn.Embedding(config.motif_size, config.n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd, padding_idx=0)
        self.ges_embedding = nn.Embedding(config.ges_size + 1, 4, padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd + 4), heads=config.n_head, seq_length=config.mutation_sampling_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)
        
        self.to_joinfeatures = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd + 4), 64),
                                       nn.ReLU())

        self.to_typefeatures = nn.Sequential(nn.Linear(64, 32),
                                       nn.ReLU())

        self.to_subtypefeatures = nn.Sequential(nn.Linear(64, 32),
                                       nn.ReLU())

        self.to_typeprobs = nn.Linear(32, config.num_class)

        self.to_subtypeprobs = nn.Linear(32, config.num_subclass)
        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x, targets=None, vis=None,get_features=False):

        triplettoken = x[:, 0, :]
        postoken = x[:, 1, :]
        gestoken = x[:, 2, :]

        tokens = self.token_embedding(triplettoken)
        positions = self.position_embedding(postoken)
        ges = self.ges_embedding(gestoken)

        x = torch.cat((tokens, positions, ges), axis=2)

        x = self.do(x)
        # pdb.set_trace()

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over the time dimension

        join_features = self.to_joinfeatures(x)

        type_features = self.to_typefeatures(join_features)

        subtype_features = self.to_subtypefeatures(join_features)

        typeprobs = self.to_typeprobs(type_features)
        subtypeprobs = self.to_subtypeprobs(subtype_features)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss1 = F.cross_entropy(typeprobs.view(-1, typeprobs.size(-1)), targets.view(-1))
            loss2 = F.cross_entropy(subtypeprobs.view(-1, subtypeprobs.size(-1)), targets.view(-1))
            loss = loss1 + loss2

        logits_feats = {'first_logits': typeprobs,
                        'second_logits': subtypeprobs,
                        'first_features': type_features,
                        'second_features': subtype_features,
                        'join_features': join_features
                        }
        return logits_feats, loss

class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)
    
    def contains_nan(self,tensor):
        return bool((tensor != tensor).sum() > 0)

    def forward(self, x,vis=False):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        #pdb.set_trace()
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits

        if vis:
            return dot

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities        

        assert not self.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)


        return self.unifyheads(out)

