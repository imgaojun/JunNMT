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
