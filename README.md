# JunNMT
Neural Machine Translation in Pytorch

### About


### Requirements
- Python >= 3.5
- Pytorch == 0.4
- torchtext
- tensorboardX
- progressbar2

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
cp JunNMT/scripts/translate.sh ./
cp JunNMT/config.yml ./
```
#### 3.Preprocessing

Edit the script file `build_vocab.sh`.

```
NMT_DIR= 
python3 ${NMT_DIR}/build_vocab.py \
    -train_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -train_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -valid_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -valid_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -save_data demo \
    -config ./config.yml
```

| parameter     | description |
|---            |--- |
| -train_src FILE |  Source Training Set |
| -train_tgt FILE |  Target Training Set |
| -valid_src FILE |  Source Validation Set |
| -valid_tgt FILE |  Target Validation Set |
| -save_data STR  |  the Prefix of Output File Name |
| -config FILE    |  Configuration File |


#### 4.Training
Edit the script `train.sh`.

```
NMT_DIR=PATH_TO_JunNMT
python3 ${NMT_DIR}/train.py \
    -gpuid 0 \
    -config ./config.yml \
    -nmt_dir ${NMT_DIR} \
    -train_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -train_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -valid_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -valid_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -vocab ./demo.vocab.pkl

```

| parameter     | description |
|---            |---          |
| -gpuid INT    |  Choose Which GPU A Program Uses |
| -config FILE  |  Configuration File |
| -nmt_dir PATH |  Path to JunNMT Directory |
| -train_src FILE |             |
| -train_tgt FILE |             |
| -valid_src FILE |             |
| -valid_tgt FILE |             |
| -vocab FILE     |             |

#### 5.Visualizing training phase
If you want to visualize your training phase, you need to install tensorflow(for tensorboard web server) first, since the projects uses tensorboard for visualizing.

After installing `tensorflow`, you can start a server by using the following command.

```
tensorboard --logdir ./${log_dir} --port 6006
```

And then you can watch your training phase on your browser.

#### 6.Testing
To perform testing, just run `sh traslate.sh`.

| parameter     | description |
|---            |--- |
| -gpuid INT    |  Choose Which GPU A Program Uses |
| -test_data FILE  |  test file |
| -test_out FILE |  output file    |
| -model FILE   |  load existing model |
| -vocab FILE     |   |
| -beam_size INT |   |
| -decode_max_length INT|   |
| -config FILE | |
| -dump_beam FILE|  Save  beam trace |
### 7.Visualizing the beam search
To visualize the beam search exploration, you can use the option -dump_beam beam.json. It will save a JSON serialization of the beam search history.
This representation can then be visualized dynamically using the generate_beam_viz.py script from the JunNMT/tools/VisTools(borrowed from Opennmt).
```
python ./JunNMT/tools/VisTools/generate_beam_viz.py -d ./beam.json -o ./beam_viz_dir -v ./demo.tgt_vocab.json
```