NMT_DIR = 
python3 ${NMT_DIR}/preprocess.py \
    -train_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -train_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -valid_src /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.cn \
    -valid_tgt /home/gaojun4ever/Documents/Projects/mt-exp/data/dev/nist02.en0 \
    -save_data demo \
    -config ./config.yml \
