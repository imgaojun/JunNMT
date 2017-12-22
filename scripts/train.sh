NMT_DIR=PATH_TO_JunNMT
python3 ${NMT_DIR}/train.py \
    -gpuid 0 \
    -config ./config.yml \
    -nmt_dir ${NMT_DIR} \
    -data demo
