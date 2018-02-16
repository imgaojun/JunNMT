# JunNMT
Neural Machine Translation in Pytorch

### About


### Requirements
- Python >= 3.5
- Pytorch == 0.30
- torchtext
- tensorboardX

### Configuration
`JunNMT/config.yml` is a configuration file, which contains configurations of model, training, and tesing.

### Qucik Start

#### 1.Start a project

```
mkdir demo
cd demo
```

#### 2.Copy scripts
Copy some useful scripts and template configuration to the project.

```
cp JunNMT/scripts/preprocess.sh ./
cp JunNMT/scripts/train.sh ./
cp JunNMT/scripts/traslate.sh ./
cp JunNMT/config.yml ./
```
#### 3.Do preprocessing

Edit the script file `preprocess.sh`.

```
NMT_DIR= 
python3 ${NMT_DIR}/preprocess.py \
    -train_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -train_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -valid_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -valid_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -save_data demo \
    -config ./config.yml
```

| parameter     | description |
|---            |--- |
| -train_src 'FILE' |  Source Training Set |
| -train_tgt 'FILE' |  Target Training Set |
| -valid_src 'FILE' |  Source Validation Set |
| -valid_tgt 'FILE' |  Target Validation Set |
| -save_data 'STR'  |  the Prefix of Output File Name |
| -config FILE    |  Configuration File |

Run the following command and This commad will prepare train data, valid data and vocab file for your project.

```
sh preprocess.sh
```


#### 4.Do training
Edit the script `train.sh`.

```
NMT_DIR=
python3 ${NMT_DIR}/train.py \
    -gpuid 0 \
    -config ./config.yml \
    -nmt_dir ${NMT_DIR} \
    -data demo
```

| parameter     | description |
|---            |---          |
| -gpuid 'INT'    |  Choose Which GPU A Program Uses |
| -config 'FILE'  |  Configuration File |
| -nmt_dir 'PATH' |  Path to JunNMT Directory |
| -data 'STR'     |  the Prefix of Data File Name |

Run the command to train a model.

```
sh train.sh
```

#### 5.Visualize training progress
If you want to visualize your training progress, you need to install tensorflow(for tensorboard web server) first, since the projects uses tensorboard for visualizing.

After installing `tensorflow`, you can start a server by using the following command.

```
tensorboard --logdir ./${log_dir} --port 6006
```

And then you can watch your training progress on your browser.

#### 6.Do testing
To perform testing, just run `sh traslate.sh`.

| parameter     | description |
|---            |--- |
| -gpuid 'INT'    |  Choose Which GPU A Program Uses |
| -src_in 'FILE'  |  test file |
| -tgt_out 'FILE' |  output file    |
| -model 'FILE'   |  load existing model |
| -data 'STR'     |  the Prefix of Data File Name |