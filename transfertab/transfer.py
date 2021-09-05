# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_transfer.ipynb (unless otherwise specified).

__all__ = ['get_metadict_skeleton', 'transferembeds_']

# Cell
import pathlib
import torch
import torch.nn as nn
import json
import pandas as pd
from functools import partial
from fastcore.foundation import *
from fastcore.dispatch import *
from nbdev.showdoc import *
from .utils import *
from .extract import *

# Cell
def get_metadict_skeleton(df: pd.DataFrame, *, catcols=None, path=None):
    catdict = getcatdict(df, catcols)
    metadict = {}
    for (cat, classes) in catdict.items():
        metadict[cat] = {'mapped_cat': '', 'classes_info': {clas: [] for clas in classes}}
    if path != None:
        with open(path, 'w') as fp:
            json.dump(metadict, fp)
    return metadict

# Cell
@typedispatch
def transferembeds_(
        dest_embeds: nn.Module,
        src_embeds: nn.Module,
        /,
        metatransfer,
        transfer_cats,
        *,
        newcatcols,
        oldcatcols,
        oldcatdict,
        newcatdict,
        aggfn = partial(torch.mean, dim=0)):
    '''
        Transfers embeddings from `src_embeds` to `dest_embeds`,
        with the help of collections containing various metadata.
    '''
    src_state_dict = L(src_embeds.state_dict().items())
    dest_state_dict = L(dest_embeds.state_dict().items())
    for newcat in transfer_cats:
        newidx = newcatcols.index(newcat)
        oldidx = oldcatcols.index(metatransfer[newcat]["mapped_cat"])
        new_ps = torch.zeros(src_state_dict[oldidx][1].shape[1], 0)
        for newclass in newcatdict[newcat]:
            classidxs = L(oldcatdict[oldcatcols[oldidx]]).argwhere(lambda x: x in metatransfer[newcat]["classes_info"][newclass])
            if len(classidxs) == 0:
                classidxs =  list(range(len(oldcatdict[oldcatcols[oldidx]])))
            ps = torch.unsqueeze(aggfn(torch.index_select(src_state_dict[oldidx][1], 0, torch.LongTensor(classidxs))), -1)
            new_ps = torch.cat((new_ps, ps), dim=1)
        dest_embeds.state_dict()[dest_state_dict[newidx][0]].copy_(new_ps.T)

@typedispatch
def transferembeds_(
        dest_embeds: nn.Module,
        src_embeds: dict,
        metatransfer,
        transfer_cats,
        *,
        newcatcols,
        oldcatcols,
        newcatdict,
        aggfn = partial(torch.mean, dim=0)):
    dest_state_dict = L(dest_embeds.state_dict().items())
    for newcat in transfer_cats:
        newidx = newcatcols.index(newcat)
        oldcatname = metatransfer[newcat]['mapped_cat']
        new_ps = torch.zeros(torch.tensor(src_embeds[oldcatname]['embeddings']).shape[1], 0)
        for newclass in newcatdict[newcat]:
            classidxs = L(src_embeds[oldcatname]['classes']).argwhere(lambda x: x in metatransfer[newcat]["classes_info"][newclass])
            if len(classidxs) == 0:
                classidxs = list(range(len(src_embeds[oldcatname]['classes'])))
            ps = torch.unsqueeze(aggfn(torch.index_select(torch.tensor(src_embeds[oldcatname]['embeddings']), 0, torch.LongTensor(classidxs))), -1)
            new_ps = torch.cat((new_ps, ps), dim=1)
        dest_embeds.state_dict()[dest_state_dict[newidx][0]].copy_(new_ps.T)


@typedispatch
def transferembeds_(
        dest_embeds: nn.Module,
        src_embeds: (pathlib.PosixPath, str),
        metatransfer,
        transfer_cats,
        *,
        kind = "bson",
        **kwargs):
    if kind == "json":
        with open(src_embeds, 'r') as fp:
            src_embeds = json.loads(fp.read())
    else:
        src_embeds = load_bson(src_embeds)
    transferembeds_(dest_embeds, src_embeds, metatransfer, transfer_cats, **kwargs);


# Cell
#nbdev_comment _all_ = ['transferembeds_']