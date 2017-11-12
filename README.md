# JunNMT
Neural Machine Translation in Pytorch

### About

### Requirements
- Python >= 3.5
- Pytorch >= 0.20
- tensorboardx

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
