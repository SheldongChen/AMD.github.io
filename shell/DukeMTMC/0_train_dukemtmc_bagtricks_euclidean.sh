#cd pth_to_fast-reid-interpretation
cd /export/home/cxd/fast-reid-interpretation-1008
#set gpus
gpus='0'
#train
CUDA_VISIBLE_DEVICES=$gpus python ./tools/train_net.py  --config-file ./configs/DukeMTMC/bagtricks_circle_R50.yml  MODEL.BACKBONE.PRETRAIN_PATH  '/export/home/pretrain_models/resnet50-19c8e357.pth'  MODEL.DEVICE "cuda:0"   MODEL.BACKBONE.WITH_NL  False   TEST.METRIC   "euclidean"   TEST.EVAL_PERIOD 10   SOLVER.CHECKPOINT_PERIOD 10

