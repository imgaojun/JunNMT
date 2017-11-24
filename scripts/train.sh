export CUDA_VISIBLE_DEVICES=0
NMT_DIR=
python3 ${NMT_DIR}/train.py \
    -gpuid 0 \
    -config ./config.yml \
    -nmt_dir ${NMT_DIR} \
    -data demo
