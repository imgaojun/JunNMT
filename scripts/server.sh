export CUDA_VISIBLE_DEVICES=0
NMT_DIR=PATH_TO_JunNMT
python3 ${NMT_DIR}/JunNMT/translate.py \
    --config ./out_dir/config.yml \
    --model ./out_dir/checkpoint_epoch1.pkl \
    --data demo