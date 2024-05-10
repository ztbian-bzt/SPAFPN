# SPAFPN: Wear a Multi-scale Feature Scarf around Neck for Real-time Object Detector
In this paper, we propose the Scarf Path Aggregation Feature Pyramid Network (SPAFPN), an advanced neck structure of multi-scale fusion for real-time object detection. SPAFPN adheres to the decentralized multi-scale fusion idea of "Light Fusion, Heavy Decouple" while inheriting the concept of modularization design. SPAFPN can promote the cross-scale low-loss transfer of features and improve the performance of the model, which mainly consists of Pyramid Fusion and Multi-Concat modules.
## Models
### Object Detection(COCO)

|Model|Size|Param.|GFLOPs|mAP<sup>val</sup><sub>50-95</sub>|FPS<sub>RTX4090,half,b32,pytorch</sub>|
|-|-|-|-|-|-|
|[SPAFPN-C2f-n](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNn-C2f.pt)|640|3.7M|10.7|41.3%|787|
|[SPAFPN-C2f-s](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNs-C2f.pt)|640|13.1M|34.9|47.3%|573|
|[SPAFPN-C2f-m](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNm-C2f.pt)|640|28.3M|89.4|50.7%|403|
|[SPAFPN-GELAN-N](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNn-GELAN.pt)|640|3.4M|10.9|42.7%|579|
|[SPAFPN-GELAN-S](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNs-GELAN.pt)|640|11.8M|35.6|48.0%|494|
|[SPAFPN-GELAN-M](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNm-GELAN.pt)|640|23.8M|81.7|51.0%|343|
|[SPAFPN-HG-N](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNn-HG.pt)|640|3.4M|11.4|42.6%|661|
|[SPAFPN-HG-S](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNs-HG.pt)|640|11.9M|37.7|48.2%|516|
|[SPAFPN-HG-M](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNm-HG.pt)|640|25.0M|90.5|51.3%|351|


### Instance segmentation(COCO)
|Model|Size|Param.|GFLOPs|mAP<sup>box</sup><sub>50-95</sub>|mAP<sup>mask</sup><sub>50-95</sub>
|-|-|-|-|-|-|
|[SPAFPN-C2f-seg-n](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNn-C2f-seg.pt)|640|4.0M|14.6|41.1%|34.6%|
|[SPAFPN-C2f-seg-s](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNs-C2f-seg.pt)|640|13.8M|48.9|47.4%|39.8%|
|[SPAFPN-C2f-seg-m](https://github.com/ztbian-bzt/SPAFPN-assets/releases/download/v0.0.0-beta/SPAFPNm-C2f-seg.pt)|640|29.7M|120.7|50.7%|42.4%|


## Environment and Datasets
- Environment

Create environmemt. 

Make sure the torch version matches cuda.
We use and recommend python 3.8,  torch 2.1.1, torchvision 0.16.1, torchaudio 0.9.0 and cuda 11.8.
The torch installation command we used is `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`. If you want to use the same torch, uncomment the third line of setup.sh.

```
conda create --name any_name_you_want python=3.8
conda activate any_name_you_want
cd .../.../SPAFPN-main
sh setup.sh
```
- Datasets

Place the dataset as shown in `SPAFPN-main/ultralytics/cfg/datasets/coco.yaml` and `SPAFPN-main/ultralytics/cfg/datasets/coco-seg.yaml`.

And convert it to YOLO format by **COCO2YOLO.py**.


## Train

## Evaluate



