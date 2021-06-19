# transfertab
> Allow transfer learning using structured data.


## Install

`pip install transfertab`

## How to use

We have two fastai learners `learn1` and `learn2`  created, with models having embedding layers for their categorical variables.

```
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



For transferring the embeddings from `learn1` to `learn2`, we first instantiate the `TabTransfer` class with the learners and the categorical variables to transfer.

```
tabobj = TabTransfer(learn2)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-7bae12932fef> in <module>
    ----> 1 tabobj = TabTransfer(learn2)
    

    NameError: name 'TabTransfer' is not defined


Now, we can just call the transfer function to start the process.

```
tabobj.init_from_json("../data/adults.json")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-a78738acd2cf> in <module>
    ----> 1 tabobj.init_from_json("../data/adults.json")
    

    NameError: name 'tabobj' is not defined


```
tabobj.transfer(verbose = True)
```

    mean is tensor([ 0.0027, -0.0050, -0.0009,  0.0026,  0.0011, -0.0004],
           grad_fn=<MeanBackward1>) for Parameter containing:
    tensor([[-0.0156, -0.0057,  0.0052,  0.0033,  0.0026,  0.0061],
            [-0.0070, -0.0103, -0.0070,  0.0056,  0.0013,  0.0010],
            [ 0.0118, -0.0094, -0.0015, -0.0127, -0.0066, -0.0163],
            [ 0.0042, -0.0039, -0.0106, -0.0068, -0.0124, -0.0183],
            [-0.0010, -0.0093, -0.0049,  0.0077,  0.0119, -0.0080],
            [-0.0022, -0.0082,  0.0072,  0.0114,  0.0166,  0.0077],
            [-0.0007,  0.0102,  0.0074, -0.0060, -0.0025, -0.0020],
            [ 0.0077, -0.0047,  0.0007,  0.0156,  0.0053,  0.0077],
            [ 0.0181, -0.0021,  0.0005,  0.0061, -0.0098,  0.0119],
            [ 0.0113, -0.0067, -0.0055,  0.0019,  0.0041,  0.0066]],
           requires_grad=True)
    0, <class 'int'>
    Transferring weights for class #na#, cat workclass from previous weights
    old weight for class is tensor([ 0.0068, -0.0076,  0.0119,  0.0084,  0.0189,  0.0128],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0156, -0.0057,  0.0052,  0.0033,  0.0026,  0.0061],
           grad_fn=<SliceBackward>)
    1, <class 'int'>
    Transferring weights for class  ?, cat workclass from previous weights
    old weight for class is tensor([ 0.0055,  0.0081, -0.0086, -0.0007, -0.0017,  0.0016],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0070, -0.0103, -0.0070,  0.0056,  0.0013,  0.0010],
           grad_fn=<SliceBackward>)
    2, <class 'int'>
    Transferring weights for class  Federal-gov, cat workclass from previous weights
    old weight for class is tensor([-0.0093, -0.0077, -0.0013,  0.0005, -0.0187, -0.0039],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0118, -0.0094, -0.0015, -0.0127, -0.0066, -0.0163],
           grad_fn=<SliceBackward>)
    3, <class 'int'>
    Transferring weights for class  Local-gov, cat workclass from previous weights
    old weight for class is tensor([ 0.0048, -0.0019, -0.0013, -0.0093,  0.0103, -0.0037],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0042, -0.0039, -0.0106, -0.0068, -0.0124, -0.0183],
           grad_fn=<SliceBackward>)
    4, <class 'int'>
    Transferring weights for class  Never-worked, cat workclass from previous weights
    old weight for class is tensor([-0.0015,  0.0175,  0.0141,  0.0016, -0.0165,  0.0019],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0010, -0.0093, -0.0049,  0.0077,  0.0119, -0.0080],
           grad_fn=<SliceBackward>)
    5, <class 'int'>
    Transferring weights for class  Private, cat workclass from previous weights
    old weight for class is tensor([-0.0078, -0.0189,  0.0027, -0.0117, -0.0189,  0.0057],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0022, -0.0082,  0.0072,  0.0114,  0.0166,  0.0077],
           grad_fn=<SliceBackward>)
    6, <class 'int'>
    Transferring weights for class  Self-emp-inc, cat workclass from previous weights
    old weight for class is tensor([ 0.0014, -0.0126,  0.0135,  0.0033,  0.0059,  0.0031],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0007,  0.0102,  0.0074, -0.0060, -0.0025, -0.0020],
           grad_fn=<SliceBackward>)
    7, <class 'int'>
    Transferring weights for class  Self-emp-not-inc, cat workclass from previous weights
    old weight for class is tensor([-0.0043, -0.0021, -0.0005, -0.0076, -0.0050, -0.0083],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0077, -0.0047,  0.0007,  0.0156,  0.0053,  0.0077],
           grad_fn=<SliceBackward>)
    8, <class 'int'>
    Transferring weights for class  State-gov, cat workclass from previous weights
    old weight for class is tensor([ 0.0187, -0.0121,  0.0022, -0.0013,  0.0056, -0.0176],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0181, -0.0021,  0.0005,  0.0061, -0.0098,  0.0119],
           grad_fn=<SliceBackward>)
    9, <class 'int'>
    Transferring weights for class  Without-pay, cat workclass from previous weights
    old weight for class is tensor([-0.0038,  0.0069,  0.0018, -0.0066, -0.0016, -0.0106],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0113, -0.0067, -0.0055,  0.0019,  0.0041,  0.0066],
           grad_fn=<SliceBackward>)
    10, <class 'int'>
    Transferring weights for class Private, cat workclass using mean
    old weight for class is tensor([-0.0024,  0.0142, -0.0060, -0.0050, -0.0119, -0.0003],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0027, -0.0050, -0.0009,  0.0026,  0.0011, -0.0004],
           grad_fn=<SliceBackward>)

