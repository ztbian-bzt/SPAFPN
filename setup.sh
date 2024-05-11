#!/usr/bin/env bash
# please cd to SPAFPN-main
# pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
mim install mmcv>=1.5.0
pip install mmdet>=2.28.1
cd ultralytics/nn/modules/dcn/ops_dcnv3
sh make.sh
#cd ../DCNv4_op # optional for DCNv4
#sh make.sh
