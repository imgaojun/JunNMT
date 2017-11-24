export CUDA_VISIBLE_DEVICES=0
NMT_DIR=
python3 ${NMT_DIR}/JunNMT/translate.py \
    --config ./config.yml \
    --src_in /home/xiapeng/gaojun/trans/test/nist02.cn \
    --tgt_out ./test0_out1 \
    --model /home/xiapeng/gaojun/JunNMT_test/nmt_test/lstm_test/test0/checkpoint_epoch1.pkl

perl ../scripts/multi-bleu.pl ../../../trans/test/nist02.en0 ../../../trans/test/nist02.en1 ../../../trans/test/nist02.en2 ../../../trans/test/nist02.en3 < test0_out1