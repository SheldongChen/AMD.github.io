cd /export/home/cxd/fast-reid-interpretation-1008

gpus='1'

CUDA_VISIBLE_DEVICES=$gpus python ./projects/InterpretationReID/train_net.py  --config-file ./projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip.yml    MODEL.DEVICE "cuda:0"
