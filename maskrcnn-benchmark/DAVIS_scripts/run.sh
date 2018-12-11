export PYTHONPATH="${PYTHONPATH}:/home/guli/Desktop/VOS_ICCV2019/maskrcnn-benchmark

python davis_baseline_test.py --config-file /home/guli/Desktop/VOS_ICCV2019/maskrcnn-benchmark/configs/davis/e2e_mask_rcnn_R_50_FPN_1x_davis_test.yaml


export NGPUS=2
python  -m torch.distributed.launch --nproc_per_node=$NGPUS davis_baseline_test.py --config-file /home/guli/Desktop/VOS_ICCV2019/maskrcnn-benchmark/configs/davis/e2e_mask_rcnn_R_50_FPN_1x_davis_test.yaml
