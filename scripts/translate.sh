export CUDA_VISIBLE_DEVICES=0
python3 ../translate.py \
    --config ../config.yml \
    --src_in /home/xiapeng/python/process_data/res/deve_src_file \
    --tgt_out ./test_out \
    --model /home/xiapeng/gaojun/JunNMT_test/gru_test1/epoch5/model.pkl