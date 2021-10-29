
cd /export/home/cxd/fast-reid-interpretation-1008

gpus='0'

CUDA_VISIBLE_DEVICES=$gpus python ./projects/InterpretationReID/train_net.py  --config-file ./projects/InterpretationReID/configs/DukeMTMC_Circle/circle_R50_ip.yml    MODEL.DEVICE "cuda:0"
