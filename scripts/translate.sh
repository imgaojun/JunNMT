export CUDA_VISIBLE_DEVICES=0
NMT_DIR=
python3 ${NMT_DIR}/JunNMT/translate.py \
    --config ./config.yml \
    --src_in /home/xiapeng/gaojun/trans/test/nist02.cn \
    --tgt_out ./test_out \
    --model ./our_dir/checkpoint_epoch1.pkl \
    --data demo

perl ../scripts/multi-bleu.pl ../../../trans/test/nist02.en0 ../../../trans/test/nist02.en1 ../../../trans/test/nist02.en2 ../../../trans/test/nist02.en3 < test_out