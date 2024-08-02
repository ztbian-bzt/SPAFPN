We express our sincere gratitude for your professional and comprehensive review. And we will provide responses to each of your concerns in the 'Weaknesses' section.

- ***The novelty of the framework is limited. The idea of SPAFPN that employing full-scale feature extraction for the FPN has been explored in the previous paper [1]. What's the difference between SPAFPN and this work?***

**Reply:** ASFF proposed in [1] performs multi-scale fusion at three scales in a weighted sum manner, and the output of ASFF is directly used as the output of Neck. In addition to the difference in the output scale, the input obtained by the three times of ASFF is completely consistent.

Our work follows the strategy of "Light Fusion, Heavy Decouple", and uses the convenient Pyramid Fusion module to fuse multi-scale features (" Light Fusion "). And the Multi-Concat module decouples multi-scale features back into individual scales (" Heavy Decouple ").

The ASFF module is similar in effect and structure to our Pyramid Fusion module (the difference is that Pyramid Fusion is more lightweight and uses concatenation to fuse multi-scale features).

In the article of YOLOv4 [2], it has been shown that the performance of ASFF is weaker than that of PAFPN [3], a subsequent proposed Neck structure. Our work SPAFPN complements PAFPN with multi-scale fusion features, and the performance is significantly higher than PAFPN (shown in Table 5). The portability of SPAFPN is also guaranteed.

Thank you for pointing out this similar work, we will add this discussion in the next version of the paper.

- ***There is a typo in line 137: "Conv-BN-SLIU" -> "Conv-BN-SiLU"***

**Reply:** Thank you for pointing out our typos. We will correct it in the next version of the paper and carefully check other typos that may appear.


*[1] Songtao Liu, Di Huang, and Yunhong Wang. Learning spatial fusion for single-shot object detection. arXiv preprint arXiv:1911.09516, 2019.*

*[2] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020.*

*[3] Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. Path aggregation network for instance segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8759–8768, 2018.*
