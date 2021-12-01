# transfertab
> Allow transfer learning using structured data.


## Install

```bash
pip install transfertab
```

## How to use

TransferTab enables effective transfer learning from models trained on tabular data.

To make use of `transfertab`, you'll need  
	* A pytorch model which contains some embeddings in a layer group.  
	* Another model to transfer these embeddings to, along with the metadata about the dataset on which this model will be trained.

The whole process takes place in two main steps-  
	1. Extraction  
	2. Transfer


### Extraction
This involves storing the embeddings present in the model to a `JSON` structure. This `JSON` would contain the embeddings related to the categorical variables, and can be later transfered to another model which can also benefit from these categories. It will also be possible to have multiple `JSON` files constructed from various models with different categorical variables and then use them together.

Here we'll quickly construct a `ModuleList` with a bunch of `Embedding` layers, and see how to transfer it's embeddings.

```
emb_szs1 = ((3, 10), (2, 8))
emb_szs2 = ((2, 10), (2, 8))
```

```
embed1 = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs1])
embed2 = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs2])
```

```
embed1
```




    ModuleList(
      (0): Embedding(3, 10)
      (1): Embedding(2, 8)
    )



We can call the `extractembeds` function to extract the embeddings. Take a look at the documentation to see other dispatch methods, and details on the parameters.

```
df = pd.DataFrame({"old_cat1": [1, 2, 3, 4, 5], "old_cat2": ['a', 'b', 'b', 'b', 'a'], "old_cat3": ['A', 'B', 'B', 'B', 'A']})
cats = ("old_cat2", "old_cat3")
embdict = extractembeds(embed2, df, transfercats=cats, allcats=cats)
```

```
embdict
```




    {'old_cat2': {'classes': ['a', 'b'],
      'embeddings': [[-0.28762340545654297,
        -0.142189621925354,
        0.2027226686477661,
        1.1096185445785522,
        -0.4540262520313263,
        -1.346120834350586,
        0.048871781677007675,
        0.1740419715642929,
        0.002095407573506236,
        0.721653163433075],
       [-0.9072648882865906,
        2.674738645553589,
        -0.8560850024223328,
        -1.119917869567871,
        -0.19618849456310272,
        1.1431224346160889,
        -0.5177133679389954,
        -0.6497849822044373,
        -0.9011525511741638,
        0.9314191341400146]]},
     'old_cat3': {'classes': ['A', 'B'],
      'embeddings': [[2.5755045413970947,
        -1.3670053482055664,
        -0.3207620680332184,
        -1.1824427843093872,
        0.07631386071443558,
        0.501422107219696,
        0.8510317802429199,
        -0.6687257289886475],
       [-1.3658113479614258,
        -0.27968257665634155,
        0.26537612080574036,
        0.36773681640625,
        -0.9940593242645264,
        0.9408144354820251,
        0.5295664668083191,
        -0.5038257241249084]]}}



### Transfer
The transfer process involves using the extracted weights, or a model directly and reusing trained paramters. We can define how this process will take place using the `metadict` which is a mapping of all the categories (in the current dataset), and contains information about the category it is mapped to (from the previous dataset which was used to train the old model), and how the new classes map to the old classes. We can even choose to map multiple classes to a single one, and in this case the `aggfn` parameter is used to aggregate the embedding vectors.

```
json_file_path = "../data/jsons/metadict.json"

with open(json_file_path, 'r') as j:
     metadict = json.loads(j.read())
```

```
metadict
```




    {'new_cat1': {'mapped_cat': 'old_cat2',
      'classes_info': {'new_class1': ['a', 'b'],
       'new_class2': ['b'],
       'new_class3': []}},
     'new_cat2': {'mapped_cat': 'old_cat3',
      'classes_info': {'new_class1': ['A'], 'new_class2': []}}}



We take a look at the layer parameters before and after transferring to see if it worked as expected.

```
embed1.state_dict()
```




    OrderedDict([('0.weight',
                  tensor([[-0.6940, -0.0337,  0.9491, -1.0520,  0.7804,  2.0246,  0.4242, -1.8351,
                            0.4660,  1.7667],
                          [-0.2802,  0.6081, -0.8459, -0.3288, -1.1264,  0.7621,  0.9347,  1.8096,
                           -0.1998, -0.2541],
                          [ 0.5706, -0.5213, -0.1398, -0.3742, -1.1951,  1.9640,  0.4132,  2.0365,
                            0.0655,  0.5189]])),
                 ('1.weight',
                  tensor([[ 0.9506, -0.0057,  0.2754,  0.8276,  0.8675,  1.2238, -1.5603,  1.0301],
                          [-0.7315, -0.3735,  0.6059,  0.2659, -0.4918,  1.5501,  0.0221, -0.6199]]))])



```
embed2.state_dict()
```




    OrderedDict([('0.weight',
                  tensor([[-2.8762e-01, -1.4219e-01,  2.0272e-01,  1.1096e+00, -4.5403e-01,
                           -1.3461e+00,  4.8872e-02,  1.7404e-01,  2.0954e-03,  7.2165e-01],
                          [-9.0726e-01,  2.6747e+00, -8.5609e-01, -1.1199e+00, -1.9619e-01,
                            1.1431e+00, -5.1771e-01, -6.4978e-01, -9.0115e-01,  9.3142e-01]])),
                 ('1.weight',
                  tensor([[ 2.5755, -1.3670, -0.3208, -1.1824,  0.0763,  0.5014,  0.8510, -0.6687],
                          [-1.3658, -0.2797,  0.2654,  0.3677, -0.9941,  0.9408,  0.5296, -0.5038]]))])



```
transfer_cats = ("new_cat1", "new_cat2")
newcatcols = ("new_cat1", "new_cat2")
oldcatcols = ("old_cat2", "old_cat3")

newcatdict = {"new_cat1" : ["new_class1", "new_class2", "new_class3"], "new_cat2" : ["new_class1", "new_class2"]}
oldcatdict = {"old_cat2" : ["a", "b"], "old_cat3" : ["A", "B"]}

transferembeds_(embed1, embdict, metadict, transfer_cats, newcatcols=newcatcols, oldcatcols=oldcatcols, newcatdict=newcatdict)
```

```
embed1.state_dict()
```




    OrderedDict([('0.weight',
                  tensor([[-0.5974,  1.2663, -0.3267, -0.0051, -0.3251, -0.1015, -0.2344, -0.2379,
                           -0.4495,  0.8265],
                          [-0.9073,  2.6747, -0.8561, -1.1199, -0.1962,  1.1431, -0.5177, -0.6498,
                           -0.9012,  0.9314],
                          [-0.5974,  1.2663, -0.3267, -0.0051, -0.3251, -0.1015, -0.2344, -0.2379,
                           -0.4495,  0.8265]])),
                 ('1.weight',
                  tensor([[ 2.5755, -1.3670, -0.3208, -1.1824,  0.0763,  0.5014,  0.8510, -0.6687],
                          [ 0.6048, -0.8233, -0.0277, -0.4074, -0.4589,  0.7211,  0.6903, -0.5863]]))])



As we can see, the embeddings have been transferred over.
