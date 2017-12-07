export CUDA_VISIBLE_DEVICES=0
NMT_DIR=PATH_TO_JunNMT
python3 ${NMT_DIR}/JunNMT/translate.py \
    --config ./config.yml \
    --src_in /home/xiapeng/gaojun/trans/test/nist02.cn \
    --tgt_out ./test_out \
    --model ./out_dir/checkpoint_epoch1.pkl \
    --data demo

REF0=PATH_TO_REFERENCE
REF1=PATH_TO_REFERENCE
REF2=PATH_TO_REFERENCE
REF3=PATH_TO_REFERENCE
perl ${NMT_DIR}/tools/multi-bleu.pl ${REF0} ${REF1} ${REF2} ${REF3} < test_out