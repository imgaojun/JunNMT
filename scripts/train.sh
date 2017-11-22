export CUDA_VISIBLE_DEVICES=0
python3 ../train.py \
    -gpuid 0 \
    -config ../config.yml \
    -nmt_dir /home/gaojun4ever/Documents/Projects/JunNMT \
    -data demo
