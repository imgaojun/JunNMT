# JunNMT
Neural Machine Translation in Pytorch

### About


### Requirements
- Python >= 3.5
- Pytorch >= 0.20
- tensorboardx
- torchtext

### Configuration
`JunNMT/config.yml` is a configuration file, which contains configurations of model, training, and tesing.


### Training
Edit the `JunNMT/config.yml`, and excute `JunNMT/train.py` to train a model.

| parameter     | description |
|---            |--- |
| --config PATH |  model configuration (e.g config.yml) |
| --nmt_model PATH  |  JunNMT project path |


### Tesing
To perform testing, just run `python JunNMT/translate.py`.

| parameter     | description |
|---            |--- |
| --config PATH |  model configuration (e.g config.yml) |
| --src_in PATH |  test file |
| --model PATH  |  load existing model |

### Qucik Start

#### Start a project

```
mkdir demo
cd demo
```

#### Copy scripts
Copy necessary scripts to our project
```
cp JunNMT/scripts/preprocess.sh ./
cp JunNMT/scripts/train.sh ./
cp JunNMT/scripts/traslate.sh ./
```
#### Do preprocess

Edit the script `preprocess.sh`.
| parameter     | description |
|---            |--- |
| -train_src PATH |  source training file |
| -train_tgt PATH |  target training file |
| -valid_src PATH |  valid_src file |
| -valid_tgt PATH |  valid_tgt file |
| -save_data PATH |  path to save data |
| -config PATH    |  model configuration (e.g config.yml) |

#### Do training
Edit the script `train.sh`

#### Visualize training progress
If you want to visualize your training progress, you need to install tensorflow first, since the projects uses tensorboard to do visualizing.

After installing the `tensorflow`, you can start a server by the following command.

```
tensorboard --logdir ./${log_dir} --port 6006
```

And then you can watch your training progress on you browser.

#### Do testing