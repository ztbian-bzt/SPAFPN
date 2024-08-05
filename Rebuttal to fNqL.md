We express our sincere gratitude for your professional and comprehensive review. And we will provide responses to each of your concerns in the 'Weaknesses' section.

- ***The novelty of the framework is limited. The idea of SPAFPN that employing full-scale feature extraction for the FPN has been explored in the previous paper [1]. What's the difference between SPAFPN and this work?***

**Reply:** ASFF proposed in [1] performs multi-scale fusion at three scales in a weighted sum manner, and the output of ASFF is directly used as the output of Neck. In addition to the difference in the output scale, the input obtained by the three times of ASFF is completely consistent.

Our work follows the strategy of "Light Fusion, Heavy Decouple", and uses the convenient Pyramid Fusion module to fuse multi-scale features (" Light Fusion "). And the Multi-Concat module decouples multi-scale features back into individual scales (" Heavy Decouple ").

The ASFF module is similar in effect and structure to our Pyramid Fusion module and the main difference is that Pyramid Fusion is more lightweight and uses concatenation to fuse multi-scale features.

In the article of YOLOv4 [2], it has been shown that the performance of ASFF is weaker than that of PAFPN [3], a subsequent proposed Neck structure. Our work SPAFPN complements PAFPN with multi-scale fusion features, and the performance is significantly higher than PAFPN (shown in Table 5). The portability of SPAFPN is also guaranteed.

Thank you for pointing out this similar work, we will add this discussion in the next version of the paper.

- ***This paper employs deformable convolutions in the proposed CSP-DCN module and downsampling block. The deformable convolution can bring significant performance gains but increase inference latency for most detectors. What's the performance of SPAFPN without deformable convolution?***

**Reply:** Thank you for your constructive suggestions for comparing SPAFPN without deformable convolution. We have conducted the relevant experiments, and the performance comparison is as follows:

**Ablation study on deformable conv in SPAFPN**
| Model |DCN| Param. | GFLOPs | mAP<sup>val</sup><sub>50-95</sub> | FPS |
|---|---|---|---|---|---|
|SPAFPN-C2f-s|-|15.2M|37.7|46.9%|662|
|SPAFPN-C2f-s|√|13.1M|34.9|47.2%|573|

 In comparison, we replace the deformable convolution in CSP-DCN module and downsampling block with Conv and Pooling with Conv, respectively.

- ***Inference latency is an important metric for real-time detectors. Please present the inference FPS of baseline and SPAFPN in Table 2, Table 3, and Table 5 for clear comparisons.***

Thank you for pointing out that we are missing FPS data in some tables. Table 2, Table 3, and Table 5 after updating the FPS data are shown below. We will update the table in the next version of the paper.

**Table 2**
| Model | Param. | mAP<sup>box</sup><sub>50-95</sub> | mAP<sup>mask</sup><sub>50-95</sub> |FPS|
|---|---|---|---|---|
|YOLOv5-seg-N|2.0M|27.6\%|23.4\%|-|
|YOLOv5-seg-S|7.6M|37.6\%|31.7\%|-|
|YOLOv5-seg-M|22.0M|45.0\%|37.1\%|-|
|RTMDet-Ins-tiny|5.6M|40.5\%|35.4\%|647|
|RTMDet-Ins-S|10.2M|44.0\%|38.7\%|522|
|RTMDet-Ins-M|27.6M|48.8\%|42.1\%|341|
|YOLOv8-seg-N|3.4M|36.7\%|30.5\%|876|
|YOLOv8-seg-S|11.8M|44.6\%|36.8\%|676|
|YOLOv8-seg-M|27.3M|49.9\%|40.8\%|388|
|SPAFPN-C2f-seg-N|4.0M|41.1\%|34.6\%|637|
|SPAFPN-C2f-seg-S|13.8M|47.4\%|39.8\%|497|
|SPAFPN-C2f-seg-M|29.7M|50.7\%|42.4\%|338|
|SPAFPN-GELAN-seg-N|3.7M|42.4\%|35.7\%|513|
|SPAFPN-GELAN-seg-S|12.5M|48.3\%|40.7\%|430|
|SPAFPN-GELAN-seg-M|25.2M|51.3\%|42.8\%|291|
|SPAFPN-HG-seg-N|3.7M|42.7\%|36.0\%|556|
|SPAFPN-HG-seg-S|12.6M|48.5\%|40.8\%|445|
|SPAFPN-HG-seg-M|26.4M|51.6\%|43.3\%|302|

**Table 3**
| Model |Full-Node Strategy| Param. | GFLOPs | mAP<sup>val</sup><sub>50-95</sub> | mAP<sup>val</sup><sub>50</sub>|FPS |
|---|---|---|---|---|---|---|
|YOLOv8-N|-|3.2M|8.7|37.3\%|52.6\%|1115|
|SPAFPN-C2f-N|-|3.3M|9.9|39.8\%|55.6\%|930|
|SPAFPN-C2f-N|√|3.7M|10.7|41.2\%|57.3\%|787|
|YOLOv8-S|-|11.2M|28.6|44.9\%|61.8\%|767|
|SPAFPN-C2f-S|-|11.3M|31.7|46.1\%|62.3\%|706|
|SPAFPN-C2f-S|√|13.1M|34.9|47.2\%|63.7\%|573|

**Table 5**
|Backbone|Neck|Param.| GFLOPs | mAP<sup>val</sup><sub>50-95</sub> | mAP<sup>val</sup><sub>50</sub>|FPS |
|---|---|---|---|---|---|---|
|C2f-Backbone-N|PAFPN|3.2M|8.7|37.3\%|52.6\%|1115|
|C2f-Backbone-N|AFPN|2.3M|8.1|36.8\%|49.8\%|1220|
|C2f-Backbone-N|EfficientRepBiPAN|3.0M|8.4|37.1\%|52.2\%|914|
|C2f-Backbone-N|RepGFPN|3.4M|8.9|38.3\%|54.8\%|802|
|C2f-Backbone-N|GD|6.1M|10.9|39.8\%|55.6\%|720|
|C2f-Backbone-N|SPAFPN|3.7M|10.7|41.2\%|57.3\%|787|

- ***As object detection is a well-established area, improving over state-of-the-art is important. Can the SPAFPN improve current leading DETR frameworks (e.g., Deformable-DETR)?***

**Reply:** Thank you for your excellent advice. We do need to apply SPAFPN to other models, especially DETR models, to verify its generality and portability. However, Deformable-DETR is not a real-time object detection domain and it takes a long time to train (325+ hours on NVIDIA Tesla V100 GPU). The performance of Deformable-DETR is as follows:

|Model|Param.| GFLOPs | mAP<sup>val</sup><sub>50-95</sub> | mAP<sup>val</sup><sub>50</sub>|FPS |
|---|---|---|---|---|---|
|SPAFPN-C2f-M|28.3M|89.4|50.6%|67.4%|403|
|SPAFPN-HG-M|25.0M|90.5|51.2%|68.4%|351|
|Deformable DETR|40.0M|173|46.2%|65.2%|<20|

Instead, we will apply SPAFPN to RT-DETR(CVPR 2024)[5], which is a work in the field of real-time object detection. In RT-DETR, CNN-based Cross-scale Feature Fusion (CCFF) module is a PAFPN-like structure. So let's replace it with SPAFPN. At this time, SPAFPN uses CSPRepLayer (RT-DETR/rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py Line 88), a neck feature extraction module consistent with RT-DETR. 

Due to the problem of slow convergence, the pre-trained weights of the backbone network are widely used in the training of the DETR series of detectors until the latest RT-DETR. So, we provide here the pre-trained version and the trained from scratch version of RT-DETR-SPAFPN-L.






- ***There is a typo in line 137: "Conv-BN-SLIU" -> "Conv-BN-SiLU"***

**Reply:** Thank you for pointing out our typos. We will correct it in the next version of the paper and carefully check other typos that may appear.




*[1] Songtao Liu, Di Huang, and Yunhong Wang. Learning spatial fusion for single-shot object detection. arXiv preprint arXiv:1911.09516, 2019.*

*[2] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020.*

*[3] Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. Path aggregation network for instance segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8759–8768, 2018.*

*[4] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159,2020.*

*[5] Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen. Detrs beat yolos on real-time object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16965–16974, 2024.*
