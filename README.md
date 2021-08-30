# transfertab
> Allow transfer learning using structured data.


## Install

```bash
git clone www.github.com/manikyabard/transfertab
cd transfertab
pip install -e .
```

## How to use

TransferTab enables effective transfer learning from models trained on tabular data.

To make use of `transfertab`, you'll need  
	* A pytorch model which contains some embeddings in a layer group.  
	* Another model to transfer these embeddings to, along with the metadata about the dataset on which this model will be trained.

Here we'll quickly construct a `ModuleList` with a bunch of `Embedding` layers, and see how to transfer it's embeddings.

```python
# #hide
# path = untar_data(URLs.ADULT_SAMPLE)
# df1 = pd.read_csv(path/'adult.csv')
# splits1 = RandomSplitter(valid_pct=0.2)(range_of(df1))
# to1 = TabularPandas(df1, procs=[Categorify, FillMissing,Normalize],
#                    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
#                    cont_names = ['age', 'fnlwgt', 'education-num'],
#                    y_names='salary',
#                    splits=splits1)
# dls1 = to1.dataloaders(bs=64)
# learn1 = tabular_learner(dls1, metrics=accuracy)

# #We add robot to our "race" column
# new_rows = pd.DataFrame([[49,'Private',101320,'Assoc-acdm',12.0,'Married-civ-spouse','Exec-managerial','Wife','Robot','Female',0,1902,40,'United-States','>=50k'],
#                         [18,'Private',182308,'Bachelors',10.0,'Never-married','?','Own-child','Other','Male',0,0,23,'United-States','<50k']],
#                         columns=df1.columns)
# df2 = df1.copy()
# df2 = df2.append(new_rows, ignore_index=True)
# df2.tail()

# splits2 = RandomSplitter(valid_pct=0.2)(range_of(df2))
# to2 = TabularPandas(df2, procs=[Categorify, FillMissing,Normalize],
#                    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
#                    cont_names = ['age', 'fnlwgt', 'education-num'],
#                    y_names='salary',
#                    splits=splits2)
# dls2 = to2.dataloaders(bs=64)
# learn2 = tabular_learner(dls2, metrics=accuracy)


```

The model from which we want to extract embeddings is trained on a dataset with 7 categorical variables, and 3 continuous ones. It contains embeddings for each of these categorical variables.




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
    )



The whole process takes place in two main steps-  
	1. Extraction  
	2. Transfer


### Extraction
This involves storing the embeddings present in the model to a `json` structure. This `json` would contain the embeddings related to the categorical variables, and can be later transfered to another model which can also benefit from these categories. It will also be possible to have multiple `json` files constructed from various models with different categorical variables and then use them together.

To start with the Extraction process, first we need a `metadict` containing information about the dataset on which the initial model was trained on.  
For this, we can either contruct it manually, or use one of the helper functions provided in the library.

```python
# df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>Private</td>
      <td>101320</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Married-civ-spouse</td>
      <td>NaN</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>1902</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>Private</td>
      <td>236746</td>
      <td>Masters</td>
      <td>14.0</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>10520</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>96185</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Divorced</td>
      <td>NaN</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>Self-emp-inc</td>
      <td>112847</td>
      <td>Prof-school</td>
      <td>15.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>Self-emp-not-inc</td>
      <td>82297</td>
      <td>7th-8th</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>
</div>



```python
# meta = extract_meta_from_df(df1)
# meta.keys(), meta['relationship']
```




    (dict_keys(['categories', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']),
     {'classes': ['nan',
       ' Wife',
       ' Not-in-family',
       ' Unmarried',
       ' Husband',
       ' Own-child',
       ' Other-relative']})



If we want to manually define which categories we want to extract, we can do so by defining a meta dict as shown here -

```python

# meta = {
#     "categories":['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
#     "workclass": {
#         "classes": ['nan', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov',
#            ' Federal-gov', ' Local-gov', ' ?', ' Without-pay',
#            ' Never-worked'],
#     },
#     'education': {
#         "classes": ['nan', ' Assoc-acdm', ' Masters', ' HS-grad', ' Prof-school', ' 7th-8th',
#        ' Some-college', ' 11th', ' Bachelors', ' Assoc-voc', ' 10th',
#        ' 9th', ' Doctorate', ' 12th', ' 1st-4th', ' 5th-6th',
#        ' Preschool']
#     },
#     "marital-status": {
#         "classes": ['nan', ' Married-civ-spouse', ' Divorced', ' Never-married', ' Widowed',
#        ' Married-spouse-absent', ' Separated', ' Married-AF-spouse']
#     },
#     "occupation": {
#         "classes": ["nan", ' Exec-managerial', ' Prof-specialty', ' Other-service',
#        ' Handlers-cleaners', ' Craft-repair', ' Adm-clerical', ' Sales',
#        ' Machine-op-inspct', ' Transport-moving', ' ?',
#        ' Farming-fishing', ' Tech-support', ' Protective-serv',
#        ' Priv-house-serv', ' Armed-Forces']
#     },
#     "relationship": {
#         "classes": ['nan', ' Wife', ' Not-in-family', ' Unmarried', ' Husband', ' Own-child',
#        ' Other-relative']
#     },
#     "race": {
#         "classes": ['nan', ' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo',
#        ' Other']
#     }
# }
```

More information about the metadict format can be found in the docs.

Now that we have out meta dictionary, we can start extracting the embeddings, using `extractembeds`.


<h4 id="extractembeds" class="doc_header"><code>extractembeds</code><a href="https://github.com/manikyabard/transfertab/tree/master/transfertab/utils.py#L38" class="source_link" style="float:right">[source]</a></h4>

> <code>extractembeds</code>(**`model`**, **`embeddinglg`**:`str`, **`metadict`**, **`path`**)

```
model: Any pytorch model, containing a layergroup with all the embedding layers.
embeddinglg: Name of the layer group containing the embedding layers.
metadict: A dictionary containing relevant metadata. Check the format given in docs for further details.
path: Path of the json
```


```python
# emb_details = extractembeds(learn1.model, 'embeds', meta, '../data/adult.json')
# emb_details['race']
```




    {'classes': ['nan',
      ' White',
      ' Black',
      ' Asian-Pac-Islander',
      ' Amer-Indian-Eskimo',
      ' Other'],
     'embeddings': [[-0.0019180621020495892,
       -0.019080253317952156,
       0.0008181656594388187,
       0.01788151264190674],
      [-0.0052517615258693695,
       0.011697527952492237,
       -0.008258289657533169,
       0.00018684648966882378],
      [0.0018448735354468226,
       0.009865843690931797,
       -0.009563969448208809,
       -0.012679846957325935],
      [-0.0023717426229268312,
       0.002071694005280733,
       -0.0027660217601805925,
       0.006506396923214197],
      [0.005800426006317139,
       0.01546463929116726,
       0.004760770592838526,
       -0.00994172878563404],
      [-0.0028843125328421593,
       -0.009254083968698978,
       0.005498375277966261,
       -0.0025885116774588823]]}



The embeddings will be stored in a `json` file in the given `path`. Now this file can be used to transfer these embeddings to another model.

```python
# tabobj = TabTransfer(learn1)
```

Now after creating a `TabTransfer` object, we need to initialize this with either-  
1. The path of the `json` which we just constructed.
2. The directory which contains multiple `json` files constructed using the same method, but containing various embeddings for different categorical variables needed to be transferred.

```python
# #skip
# tabobj.init_from_json("../data/adults.json")
```

There might be a case where the name of the categorical variables present in the `json` might differ from the ones present in the new model's learner. For this we can use a `mapping_dict` which maps old variable names to new ones. This can be created using the `mapping` function of the object and pass it the categorical values to transfer.

```python
# #skip
# mapping_dict = tabobj.mapping(["race", "workclass", "gender"])
# mapping_dict
```

    ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'education-num_na']





    {'race': 'race', 'workclass': 'workclass', 'gender': 'sex'}




<h4 id="TabTransfer.transfer" class="doc_header"><code>TabTransfer.transfer</code><a href="https://github.com/manikyabard/transfertab/tree/master/transfertab/core.py#L62" class="source_link" style="float:right">[source]</a></h4>

> <code>TabTransfer.transfer</code>(**`cat_names_to_transfer`**, **`layergroup`**, **`mapping_dict`**, **`verbose`**=*`False`*)




As we can see, the transfer process will start after running `tabobj.transfer` function.

```python
# #skip
# tabobj.transfer(["race", "workclass", "gender"], "embeds", {"race":"race", "workclass": "workclass", "gender":"sex"}, verbose = True)
```

    mean is tensor([ 0.0041, -0.0072, -0.0091, -0.0078]) for tensor([[-0.0118, -0.0154, -0.0116, -0.0055],
            [ 0.0281, -0.0145, -0.0503, -0.0191],
            [-0.0276, -0.0304, -0.0122,  0.0127],
            [ 0.0166, -0.0077,  0.0121, -0.0323],
            [ 0.0149, -0.0247, -0.0128, -0.0176],
            [ 0.0045,  0.0495,  0.0202,  0.0148]])
    0, <class 'int'>
    Transferring weights for class #na#, cat race using mean
    old weight for class is tensor([ 0.0041, -0.0072, -0.0091, -0.0078], grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0041, -0.0072, -0.0091, -0.0078], grad_fn=<SliceBackward>)
    1, <class 'int'>
    Transferring weights for class  Amer-Indian-Eskimo, cat race from previous weights
    old weight for class is tensor([ 0.0149, -0.0247, -0.0128, -0.0176], grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0149, -0.0247, -0.0128, -0.0176], grad_fn=<SliceBackward>)
    2, <class 'int'>
    Transferring weights for class  Asian-Pac-Islander, cat race from previous weights
    old weight for class is tensor([ 0.0166, -0.0077,  0.0121, -0.0323], grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0166, -0.0077,  0.0121, -0.0323], grad_fn=<SliceBackward>)
    3, <class 'int'>
    Transferring weights for class  Black, cat race from previous weights
    old weight for class is tensor([-0.0276, -0.0304, -0.0122,  0.0127], grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0276, -0.0304, -0.0122,  0.0127], grad_fn=<SliceBackward>)
    4, <class 'int'>
    Transferring weights for class  Other, cat race from previous weights
    old weight for class is tensor([0.0045, 0.0495, 0.0202, 0.0148], grad_fn=<SliceBackward>)
    new weight for class is tensor([0.0045, 0.0495, 0.0202, 0.0148], grad_fn=<SliceBackward>)
    5, <class 'int'>
    Transferring weights for class  White, cat race from previous weights
    old weight for class is tensor([ 0.0281, -0.0145, -0.0503, -0.0191], grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0281, -0.0145, -0.0503, -0.0191], grad_fn=<SliceBackward>)
    mean is tensor([-0.0037, -0.0007,  0.0016,  0.0003,  0.0027, -0.0014]) for tensor([[-0.0125, -0.0019, -0.0070, -0.0017, -0.0015,  0.0143],
            [-0.0002, -0.0085, -0.0127, -0.0332, -0.0274, -0.0068],
            [-0.0216, -0.0208,  0.0391,  0.0481,  0.0573,  0.0519],
            [-0.0125, -0.0039,  0.0062, -0.0273, -0.0273, -0.0192],
            [-0.0232,  0.0034, -0.0116,  0.0321,  0.0317, -0.0087],
            [ 0.0282, -0.0219,  0.0100,  0.0396,  0.0077,  0.0019],
            [-0.0218, -0.0064,  0.0136,  0.0763,  0.0721,  0.0081],
            [-0.0052,  0.0014, -0.0276, -0.0481, -0.0369, -0.0343],
            [ 0.0011,  0.0267, -0.0077, -0.0506, -0.0237,  0.0001],
            [ 0.0309,  0.0252,  0.0137, -0.0322, -0.0247, -0.0210]])
    0, <class 'int'>
    Transferring weights for class #na#, cat workclass using mean
    old weight for class is tensor([-0.0037, -0.0007,  0.0016,  0.0003,  0.0027, -0.0014],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0037, -0.0007,  0.0016,  0.0003,  0.0027, -0.0014],
           grad_fn=<SliceBackward>)
    1, <class 'int'>
    Transferring weights for class  ?, cat workclass from previous weights
    old weight for class is tensor([-0.0052,  0.0014, -0.0276, -0.0481, -0.0369, -0.0343],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0052,  0.0014, -0.0276, -0.0481, -0.0369, -0.0343],
           grad_fn=<SliceBackward>)
    2, <class 'int'>
    Transferring weights for class  Federal-gov, cat workclass from previous weights
    old weight for class is tensor([ 0.0282, -0.0219,  0.0100,  0.0396,  0.0077,  0.0019],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0282, -0.0219,  0.0100,  0.0396,  0.0077,  0.0019],
           grad_fn=<SliceBackward>)
    3, <class 'int'>
    Transferring weights for class  Local-gov, cat workclass from previous weights
    old weight for class is tensor([-0.0218, -0.0064,  0.0136,  0.0763,  0.0721,  0.0081],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0218, -0.0064,  0.0136,  0.0763,  0.0721,  0.0081],
           grad_fn=<SliceBackward>)
    4, <class 'int'>
    Transferring weights for class  Never-worked, cat workclass from previous weights
    old weight for class is tensor([ 0.0309,  0.0252,  0.0137, -0.0322, -0.0247, -0.0210],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0309,  0.0252,  0.0137, -0.0322, -0.0247, -0.0210],
           grad_fn=<SliceBackward>)
    5, <class 'int'>
    Transferring weights for class  Private, cat workclass from previous weights
    old weight for class is tensor([-0.0002, -0.0085, -0.0127, -0.0332, -0.0274, -0.0068],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0002, -0.0085, -0.0127, -0.0332, -0.0274, -0.0068],
           grad_fn=<SliceBackward>)
    6, <class 'int'>
    Transferring weights for class  Self-emp-inc, cat workclass from previous weights
    old weight for class is tensor([-0.0216, -0.0208,  0.0391,  0.0481,  0.0573,  0.0519],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0216, -0.0208,  0.0391,  0.0481,  0.0573,  0.0519],
           grad_fn=<SliceBackward>)
    7, <class 'int'>
    Transferring weights for class  Self-emp-not-inc, cat workclass from previous weights
    old weight for class is tensor([-0.0125, -0.0039,  0.0062, -0.0273, -0.0273, -0.0192],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0125, -0.0039,  0.0062, -0.0273, -0.0273, -0.0192],
           grad_fn=<SliceBackward>)
    8, <class 'int'>
    Transferring weights for class  State-gov, cat workclass from previous weights
    old weight for class is tensor([-0.0232,  0.0034, -0.0116,  0.0321,  0.0317, -0.0087],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([-0.0232,  0.0034, -0.0116,  0.0321,  0.0317, -0.0087],
           grad_fn=<SliceBackward>)
    9, <class 'int'>
    Transferring weights for class  Without-pay, cat workclass from previous weights
    old weight for class is tensor([ 0.0011,  0.0267, -0.0077, -0.0506, -0.0237,  0.0001],
           grad_fn=<SliceBackward>)
    new weight for class is tensor([ 0.0011,  0.0267, -0.0077, -0.0506, -0.0237,  0.0001],
           grad_fn=<SliceBackward>)

