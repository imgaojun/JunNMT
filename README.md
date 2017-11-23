# JunNMT
Neural Machine Translation in Pytorch

### About


### Requirements
- Python >= 3.5
- Pytorch >= 0.20
- torchtext
- tensorboardx


### Configuration
`JunNMT/config.yml` is a configuration file, which contains configurations of model, training, and tesing.

### Qucik Start

#### 1.Start a project

```
mkdir demo
cd demo
```

#### 2.Copy scripts
Copy necessary scripts to the project.

```
cp JunNMT/scripts/preprocess.sh ./
cp JunNMT/scripts/train.sh ./
cp JunNMT/scripts/traslate.sh ./
cp JunNMT/config.yml ./
```
#### 3.Do preprocess

Edit the script `preprocess.sh`.

```
NMT_DIR = PATH_TO_JunNMT
python3 ${NMT_DIR}/preprocess.py \
    -train_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -train_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -valid_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -valid_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -save_data demo \
    -config ./config.yml \
```

| parameter     | description |
|---            |--- |
| -train_src PATH |  source training file |
| -train_tgt PATH |  target training file |
| -valid_src PATH |  valid_src file |
| -valid_tgt PATH |  valid_tgt file |
| -save_data STR  |  the prefix of data file name |
| -config PATH    |  configuration file |

Run the command:

```
sh preprocess.sh
```


#### 4.Do training
Edit the script `train.sh`.

```
python3 ../train.py \
    -gpuid 0 \
    -config ../config.yml \
    -nmt_dir /home/gaojun4ever/Documents/Projects/JunNMT \
    -data demo
```

| parameter     | description |
|---            |---          |
| -gpuid INT    |  choose to use which gpu |
| -config PATH  |  configuration file |
| -nmt_dir PATH |  path to JunNMT |
| -data STR     |  the prefix of data file name |

Run the command:

```
sh train.sh
```

#### 5.Visualize training progress
If you want to visualize your training progress, you need to install tensorflow first, since the projects uses tensorboard for visualizing.

After installing the `tensorflow`, you can start a server by using the following command.

```
tensorboard --logdir ./${log_dir} --port 6006
```

And then you can watch your training progress on you browser.

#### 6.Do testing
To perform testing, just run `python JunNMT/translate.py`.

| parameter     | description |
|---            |--- |
| --config PATH |  model configuration (e.g config.yml) |
| --src_in PATH |  test file |
| --model PATH  |  load existing model |