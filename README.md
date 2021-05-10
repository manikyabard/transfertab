# transfertab
> Allow transfer learning using structured data.


## Install

`pip install transfertab`

## How to use

We have two fastai learners `learn1` and `learn2`  created, with models having embedding layers for their categorical variables.

```python
learn1, learn2, learn1.model
```




    (<fastai.tabular.learner.TabularLearner at 0x7f85da8855b0>,
     <fastai.tabular.learner.TabularLearner at 0x7f85da894130>,
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

```python
tabobj = TabTransfer(learn1, learn2)
```

Now, we can just call the transfer function to start the process.

```python
tabobj.transfer(["workclass"], verbose = True)
```

    mean is tensor([ 0.0015,  0.0052,  0.0007, -0.0013,  0.0016,  0.0003],
           grad_fn=<MeanBackward1>) for Parameter containing:
    tensor([[ 4.0982e-03,  1.0773e-02, -6.1128e-03, -1.2939e-02,  1.3693e-04,
              9.5658e-03],
            [-7.3064e-04, -9.0744e-03, -4.1995e-05,  3.8860e-03, -1.0992e-03,
             -9.0773e-03],
            [ 5.7799e-03,  1.1568e-03,  1.1390e-03,  6.0049e-03, -8.6575e-03,
              5.0584e-03],
            [ 6.2118e-03,  1.6866e-02,  1.3943e-02,  5.0229e-03,  1.2746e-02,
             -1.8922e-02],
            [-3.2620e-04,  6.3912e-03,  1.3365e-02,  1.4048e-02,  1.0076e-02,
              9.1949e-03],
            [-9.5517e-03,  7.0458e-03, -5.8635e-03, -5.2392e-03,  2.6704e-03,
              5.6363e-03],
            [ 4.7074e-03,  8.2458e-03, -1.2580e-02, -4.7994e-03,  3.7497e-03,
              7.2982e-04],
            [-7.3861e-03,  8.3785e-03, -4.2067e-03, -7.4433e-04, -3.3074e-03,
              5.5304e-03],
            [ 8.3096e-03,  7.7894e-03,  1.2156e-02, -9.9204e-03, -8.1630e-03,
             -1.5793e-02],
            [ 3.5186e-03, -5.5515e-03, -4.3965e-03, -8.7286e-03,  8.0000e-03,
              1.0594e-02]], requires_grad=True)
    0, <class 'int'>
    Transferring weights for class #na#, cat workclass from previous weights
    old weight for class is tensor([-0.0019,  0.0171, -0.0003, -0.0035, -0.0027, -0.0086],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0041,  0.0108, -0.0061, -0.0129,  0.0001,  0.0096],
           grad_fn=<SliceBackward>)
    1, <class 'int'>
    Transferring weights for class  ?, cat workclass from previous weights
    old weight for class is tensor([ 0.0045, -0.0049, -0.0006,  0.0003, -0.0065,  0.0051],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-7.3064e-04, -9.0744e-03, -4.1995e-05,  3.8860e-03, -1.0992e-03,
            -9.0773e-03], grad_fn=<SliceBackward>)
    2, <class 'int'>
    Transferring weights for class  Federal-gov, cat workclass from previous weights
    old weight for class is tensor([-0.0011,  0.0095, -0.0118,  0.0087,  0.0055,  0.0135],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0058,  0.0012,  0.0011,  0.0060, -0.0087,  0.0051],
           grad_fn=<SliceBackward>)
    3, <class 'int'>
    Transferring weights for class  Local-gov, cat workclass from previous weights
    old weight for class is tensor([ 0.0118, -0.0051, -0.0043,  0.0038,  0.0053, -0.0153],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0062,  0.0169,  0.0139,  0.0050,  0.0127, -0.0189],
           grad_fn=<SliceBackward>)
    4, <class 'int'>
    Transferring weights for class  Never-worked, cat workclass from previous weights
    old weight for class is tensor([ 0.0082,  0.0048, -0.0008, -0.0186, -0.0075,  0.0088],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0003,  0.0064,  0.0134,  0.0140,  0.0101,  0.0092],
           grad_fn=<SliceBackward>)
    5, <class 'int'>
    Transferring weights for class  Private, cat workclass from previous weights
    old weight for class is tensor([-0.0200, -0.0065, -0.0031,  0.0005,  0.0065, -0.0047],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0096,  0.0070, -0.0059, -0.0052,  0.0027,  0.0056],
           grad_fn=<SliceBackward>)
    6, <class 'int'>
    Transferring weights for class  Self-emp-inc, cat workclass from previous weights
    old weight for class is tensor([-0.0037, -0.0036,  0.0070, -0.0162,  0.0028,  0.0102],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0047,  0.0082, -0.0126, -0.0048,  0.0037,  0.0007],
           grad_fn=<SliceBackward>)
    7, <class 'int'>
    Transferring weights for class  Self-emp-not-inc, cat workclass from previous weights
    old weight for class is tensor([ 0.0197,  0.0070, -0.0086, -0.0099,  0.0142, -0.0012],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0074,  0.0084, -0.0042, -0.0007, -0.0033,  0.0055],
           grad_fn=<SliceBackward>)
    8, <class 'int'>
    Transferring weights for class  State-gov, cat workclass from previous weights
    old weight for class is tensor([ 0.0102, -0.0081, -0.0066, -0.0098,  0.0144, -0.0045],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0083,  0.0078,  0.0122, -0.0099, -0.0082, -0.0158],
           grad_fn=<SliceBackward>)
    9, <class 'int'>
    Transferring weights for class  Without-pay, cat workclass from previous weights
    old weight for class is tensor([ 0.0078, -0.0126,  0.0058,  0.0002,  0.0016, -0.0054],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0035, -0.0056, -0.0044, -0.0087,  0.0080,  0.0106],
           grad_fn=<SliceBackward>)
    10, <class 'int'>
    Transferring weights for class Private, cat workclass using mean
    old weight for class is tensor([ 0.0004,  0.0027, -0.0003, -0.0129, -0.0180,  0.0075],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0015,  0.0052,  0.0007, -0.0013,  0.0016,  0.0003],
           grad_fn=<SliceBackward>)

