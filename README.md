# SPAFPN: Wear a Multi-scale Feature Scarf around Neck for Real-time Object Detector
In this paper, we propose the Scarf Path Aggregation Feature Pyramid Network (SPAFPN), an advanced neck structure of multi-scale fusion for real-time object detection. SPAFPN adheres to the decentralized multi-scale fusion idea of "Light Fusion, Heavy Decouple" while inheriting the concept of modularization design. SPAFPN can promote the cross-scale low-loss transfer of features and improve the performance of the model, which mainly consists of Pyramid Fusion and Multi-Concat modules.
## Models
### Object Detection(COCO)

|Model|Size|Param.| GFLOPs |mAP<sup>val</sup><sub>50-95</sub>|FPS<sub>RTX4090,half,b32,pytorch</sub>|
|---|---|---|---|---|---|
|[SPAFPN-C2f-n](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNn-C2f.pt)|640|3.7M| 10.7   |41.3%|787|
|[SPAFPN-C2f-s](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNs-C2f.pt)|640|13.1M| 34.9   |47.3%|573|
|[SPAFPN-C2f-m](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNm-C2f.pt)|640|28.3M| 89.4   |50.7%|403|
|[SPAFPN-GELAN-N](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNn-GELAN.pt)|640|3.4M| 10.9   |42.7%|579|
|[SPAFPN-GELAN-S](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNs-GELAN.pt)|640|11.8M| 35.6   |48.0%|494|
|[SPAFPN-GELAN-M](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNm-GELAN.pt)|640|23.8M| 81.7   |51.0%|343|
|[SPAFPN-HG-N](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNn-HG.pt)|640|3.4M| 11.4   |42.6%|661|
|[SPAFPN-HG-S](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNs-HG.pt)|640|11.9M| 37.7   |48.2%|516|
|[SPAFPN-HG-M](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNm-HG.pt)|640|25.0M| 90.5   |51.3%|351|


### Instance segmentation(COCO)
|Model|Size|Param.|GFLOPs|mAP<sup>box</sup><sub>50-95</sub>| mAP<sup>mask</sup><sub>50-95</sub> |
|---|---|---|---|---|---|
|[SPAFPN-C2f-seg-n](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNn-C2f-seg.pt)|640|4.0M|14.6|41.1%| 34.6%                              |
|[SPAFPN-C2f-seg-s](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNs-C2f-seg.pt)|640|13.8M|48.9|47.4%| 39.8%                              |
|[SPAFPN-C2f-seg-m](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNm-C2f-seg.pt)|640|29.7M|120.7|50.7%| 42.4%                              |


## Environment and Datasets
- Environment

Create environmemt. 

Make sure the **torch version matches cuda**.
We use and recommend python 3.8,  torch 2.1.1, torchvision 0.16.1, torchaudio 0.9.0 and cuda 11.8.
The torch installation command we used is `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`. If you want to use the same torch, uncomment the third line of setup.sh.

```
conda create --name spafpn python=3.8
conda activate spafpn
cd .../.../SPAFPN-main
sh setup.sh
```
- Datasets

Download the [Microsoft COCO2017](http://cocodataset.org) datasets

Place the dataset as shown in `SPAFPN-main/ultralytics/cfg/datasets/coco.yaml` and `SPAFPN-main/ultralytics/cfg/datasets/coco-seg.yaml`.

And convert its labels for object detection to YOLO format by **COCO2YOLO.py**. Run twice to get the labels for train and val.

A tiny dataset([coco128](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/coco128.rar)) for quick start.


## Train
All our models train on 2 Nvidia RTX4090 GPUs. Our batch size is limited to 128, 64, and 48 for nano, small, and medium sizes of the model during training. The only exception is that the batch size of SPAFPN-HG-M is set to 32. If conditions permit, it is recommended to use the larger **batch** which is up to 128 and the multiples of 8. 

- Single GPU Train
```
# [] for optional. det/seg
[CUDA_VISIBLE_DEVICES=X] python train.py --data coco.yaml/coco-seg.yaml --yaml SPAFPNn-C2f.yaml/SPAFPNn-C2f-seg.yaml --batch 64 --device X [--resume XXX.pt]
```
- Multiple GPUs Train
```
# [] for optional. det/seg
[CUDA_VISIBLE_DEVICES=0,1[,2,3,4,5,6,7]] torchrun --nproc_per_node=X  [--master_port='29501'] train.py --data coco.yaml/coco-seg.yaml --yaml SPAFPNn-C2f.yaml/SPAFPNn-C2f-seg.yaml --batch 128 --device 0,1[,2,3,4,5,6,7] [--resume XXX.pt]
```

## Evaluate
- val
```
[CUDA_VISIBLE_DEVICES=X] python val.py --data coco.yaml/coco-seg.yaml --weight weights/SPAFPNn-C2f.pt --batch 32
```  
- fps(half)
```
[CUDA_VISIBLE_DEVICES=X] python val.py --data coco.yaml/coco-seg.yaml --weight weights/SPAFPNn-C2f.pt --batch 32 --half True
```  
- val(softnms)
```
[CUDA_VISIBLE_DEVICES=X] python val.py --data coco.yaml/coco-seg.yaml --weight weights/SPAFPNn-C2f.pt --batch 32 --softnms True
```

## Predict
XXXX.XXX can be an image or a folder containing images.
```
python predict.py --source XXXX.XXX --weight weights/SPAFPNn-C2f.pt
```

## Other tools
- YOLO2COCO.py

Convert YOLO format labels to COCO format.

- get_metrice.py

Get COCO/TIDE metrice. Require `pip install pycocotools>=2.0.6 tidecv>=1.0.1`.

- heatmap.py

Get Heatmap. Require `pip install Pillow>=10.0.1 grad_cam>=1.5.0`

- module_profile.py

Calculate the indicators of the module.

- COCO2YOLO-seg.py

Get the coco-seg dataset labels.

## Acknowledgements
The implementation is based on [Ultralytics](https://github.com/ultralytics/ultralytics) and also borrow some codes from below projects. Thanks for their great open-source codes.

<details><summary>others</summary>

* [DCN](https://github.com/OpenGVLab/InternImage/tree/master/detection)
* [HGBlock](https://github.com/PaddlePaddle/PaddleClas)
* [GElan](https://github.com/WongKinYiu/yolov9)
* [Dysample](https://github.com/tiny-smart/dysample)

</details>
