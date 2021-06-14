# transfertab
> Allow transfer learning using structured data.


## Install

`pip install transfertab`

## How to use

We have two fastai learners `learn1` and `learn2`  created, with models having embedding layers for their categorical variables.

```python
learn1, learn2, learn1.model
```




    (<fastai.tabular.learner.TabularLearner at 0x7fa113e953a0>,
     <fastai.tabular.learner.TabularLearner at 0x7fa113e97160>,
     TabularModel(
       (embeds): ModuleList(
         (0): Embedding(10, 6)
         (1): Embedding(17, 8)
         (2): Embedding(8, 5)
         (3): Embedding(16, 8)
         (4): Embedding(7, 5)
         (5): Embedding(6, 4)
         (6): Embedding(3, 3)
       )
       (emb_drop): Dropout(p=0.0, inplace=False)
       (bn_cont): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (layers): Sequential(
         (0): LinBnDrop(
           (0): Linear(in_features=42, out_features=200, bias=False)
           (1): ReLU(inplace=True)
           (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         )
         (1): LinBnDrop(
           (0): Linear(in_features=200, out_features=100, bias=False)
           (1): ReLU(inplace=True)
           (2): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         )
         (2): LinBnDrop(
           (0): Linear(in_features=100, out_features=2, bias=True)
         )
       )
     ))



For transferring the embeddings from `learn1.model` to `learn2`, we first extract the embeddings from the first model, save those to a json, and then use the json to finally to transfer the embeddings.
