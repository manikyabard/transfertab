# transfertab
> Allow transfer learning using structured data.


```
# #hide
# from transfertab.core import *
# from nbdev.showdoc import *
# from fastai.tabular.all import *
# from transfertab.utils import *
```

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

```
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

The whole process takes place in two main steps-  
	1. Extraction  
	2. Transfer


### Extraction
This involves storing the embeddings present in the model to a `json` structure. This `json` would contain the embeddings related to the categorical variables, and can be later transfered to another model which can also benefit from these categories. It will also be possible to have multiple `json` files constructed from various models with different categorical variables and then use them together.

To start with the Extraction process, first we need a `metadict` containing information about the dataset on which the initial model was trained on.  
For this, we can either contruct it manually, or use one of the helper functions provided in the library.

```
# df1.head()
```

```
# meta = extract_meta_from_df(df1)
# meta.keys(), meta['relationship']
```

If we want to manually define which categories we want to extract, we can do so by defining a meta dict as shown here -

```

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

```
# emb_details = extractembeds(learn1.model, 'embeds', meta, '../data/adult.json')
# emb_details['race']
```

The embeddings will be stored in a `json` file in the given `path`. Now this file can be used to transfer these embeddings to another model.

```
# tabobj = TabTransfer(learn1)
```

Now after creating a `TabTransfer` object, we need to initialize this with either-  
1. The path of the `json` which we just constructed.
2. The directory which contains multiple `json` files constructed using the same method, but containing various embeddings for different categorical variables needed to be transferred.

```
# #skip
# tabobj.init_from_json("../data/adults.json")
```

There might be a case where the name of the categorical variables present in the `json` might differ from the ones present in the new model's learner. For this we can use a `mapping_dict` which maps old variable names to new ones. This can be created using the `mapping` function of the object and pass it the categorical values to transfer.

```
# #skip
# mapping_dict = tabobj.mapping(["race", "workclass", "gender"])
# mapping_dict
```

As we can see, the transfer process will start after running `tabobj.transfer` function.

```
# #skip
# tabobj.transfer(["race", "workclass", "gender"], "embeds", {"race":"race", "workclass": "workclass", "gender":"sex"}, verbose = True)
```
