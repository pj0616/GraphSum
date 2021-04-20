# GraphSum
HiGraph include the implement of GAT and GCN
Higraph2 include the implement of GAE


## Data

Please download **CNN/DailyMail**, **Multi-News** datasets, the json-format datasets in [this link](https://drive.google.com/open?id=1JW033KefyyoYUKUFj6GqeBFZSHjksTfr)

The example looks like this:

```
{
  "text":["deborah fuller has been banned from keeping animals ... 30mph",...,"a dog breeder and exhibitor... her dogs confiscated"],
  "summary":["warning : ... at a speed of around 30mph",... ,"she was banned from ... and given a curfew "],
  "label":[1,3,6]
}
```
And put the data in '../data' file folder
After getting the standard json format, you can prepare the dataset for the graph by ***PrepareDataset.sh*** in the project directory. The processed files will be put under the ***cache*** directory.
You can also get the preprocessed TF-IDF features used in the graph creation from [here](https://drive.google.com/open?id=1oIYBwmrB9_alzvNDBtsMENKHthE9SW9z) provided by papper "Heterogeneous Graph Neural Networks for Extractive Document Summarization".


## Training

Some training parameters set example on model GCN

```
 args = parser.parse_args(['--cuda','--data_dir', '../data/multinews', '--cache_dir', '../data/cache/MultiNews', '--log_root', '../data/log/',
'--batch_size', '32', '--model_name', 'GCN',
 '--embedding_path', '../data/glove.6B/glove.6B.300d.txt', '--model', 'HSG', '--save_root', '../data/save1/'])

```

Some training parameters set example on model GAT
```
args = parser.parse_args(['--cuda','--data_dir', '../data/multinews', '--cache_dir', '../data/cache/MultiNews', '--log_root', '../data/log/',
'--batch_size', '32', '--model_name', 'GAT',
  '--embedding_path', '../data/glove.6B/glove.6B.300d.txt', '--model', 'HSG', '--save_root', '../data/save2/'])
```

Some training parameters set example on model GAE
```
args = parser.parse_args(['--cuda','--data_dir', '../data/multinews', '--cache_dir', '../data/cache/MultiNews', '--log_root', '../data/log/',
'--batch_size', '32', '--model_name', 'GAE',
  '--embedding_path', '../data/glove.6B/glove.6B.300d.txt', '--model', 'HSG2', '--save_root', '../data/save3/', ''])
```

## Test

The Test parameters set example model GCN

```
args = parser.parse_args(['--cuda','--data_dir', '../data/multinews', '--cache_dir', '../data/cache/MultiNews', '--log_root', '../data/log/',
'--batch_size', '32', '--model_name', 'GCN',
 '--embedding_path', '../data/glove.6B/glove.6B.300d.txt', '--model', 'HSG', '--save_root', '../data/save1/', '--test_model', 'multi', '--use_pyrouge'])
```

The Test parameters set example model GAT

```
 args = parser.parse_args(['--cuda','--data_dir', '../data/multinews', '--cache_dir', '../data/cache/MultiNews', '--log_root', '../data/log/',
'--batch_size', '32', '--model_name', 'GAT',
 '--embedding_path', '../data/glove.6B/glove.6B.300d.txt', '--model', 'HSG', '--save_root', '../data/save2/', '--test_model', 'multi', '--use_pyrouge'])
```

The Test parameters set example model GAE

```
args = parser.parse_args(['--cuda','--data_dir', '../data/multinews', '--cache_dir', '../data/cache/MultiNews', '--log_root', '../data/log/',
'--batch_size', '32', '--model_name', 'GAE',
  '--embedding_path', '../data/glove.6B/glove.6B.300d.txt', '--model', 'HSG2', '--save_root', '../data/save3/', '--test_model', 'multi', '--use_pyrouge'])
```