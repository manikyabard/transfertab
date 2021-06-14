# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_ExractUtils.ipynb (unless otherwise specified).

__all__ = ['extract_meta_from_learner', 'extract_meta_from_df', 'extractembeds']

# Cell
from fastai.tabular.all import *
from copy import deepcopy
import json

# Cell

#Meta Format: {'categories':['cat1','cat2'......],'cat1':{'classes':['class1','class2'......]},'cat2':...........}
def extract_meta_from_learner(learner):
    cat_list=learn.dls.cat_names
    #print(cat_list)
    temp_meta=dict.fromkeys(cat_list)
    t=learner.dls.classes
    meta={'categories':cat_list}
    for i in t:
        temp_meta[i]={'classes':t[i]}
    meta.update(temp_meta)
    return meta

def extract_meta_from_df(df: pd.DataFrame):
    columns = [x for x in df.columns]
    cat_list = []
    for i,j in enumerate(df.dtypes):
        if j == 'object':
            cat_list.append(columns[i])
    cat_dict = dict.fromkeys(cat_list)
    meta={'categories':cat_list}
    for i in cat_dict:
        cat_dict[i] = {}
        cat_dict[i]["classes"] = list(df[i].unique())
    meta.update(cat_dict)
    return meta

# Cell
def extractembeds(model, embeddinglg: str, metadict, path):
    '''
    model: Any pytorch model, containing a layergroup with all the embedding layers.
    embeddinglg: Name of the layer group containing the embedding layers.
    metadict: A dictionary containing relevant metadata. Check the format given in docs for further details.
    path: Path of the json
    '''
    embedsdict = deepcopy(metadict)
    for i, cat in enumerate(metadict["categories"]):
        try:
            classes = metadict[cat]["classes"]
            layer = getattr(model, embeddinglg)[i]
            assert (layer.num_embeddings == len(classes)), "Embeddings should have same number of classes. Something might have gone wrong."
            embedsdict[cat]["embeddings"] = layer.weight.cpu().detach().numpy().tolist()
        except KeyError:
            pass
    with open(path, 'w') as fp:
        json.dump(embedsdict, fp)
    return embedsdict