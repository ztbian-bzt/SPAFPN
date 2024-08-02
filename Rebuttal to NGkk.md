We express our sincere gratitude for your professional and comprehensive review. And we will provide responses to each of your questions in Section 'Weaknesses'.

- ***The proposed method has a low gain (0.4) on large models, which means that spafpn is not necessary when the backbone features are strong enough.***

**Reply:** It is relatively common for real-time object detection to achieve small gains for common datasets (e.g., COCO) on large models, especially when improving only one of the backbone, neck, and head items of real-time object detection. 

For example, the Gold-YOLO [1] series uses YOLOv6 3.0 [2] as its baseline model, and its gains in N/S/M size are 2.6\%/1.1\%/0.7\% mAP, respectively. Compared to SOTA YOLOv8 [3] at that time, its gain was only 2.3\%/0.5\%/-0.4\% mAP. Moreover, in February this year, the open-source YOLOv9 series [4] had a gain of only 1.8\%/0.9\% mAP in S/M size compared with YOLOv8. This is the final gain of YOLOv9 after improving the backbone, neck, and head for real-time object detection.

Our method has a gain of 3.9\%/2.3\%/0.4\% mAP in N/S/M size over YOLOv8 when only changing the neck part and can achieve a gain of 5.2\%/3.2\%/1.1\% mAP when also changing the backbone part.

- ***The author claims that cross-scale interactions can help detect small and medium objects(L121-122), but this is not supported by experiments.***

**Reply:** The sentence in L121-122 is `This reduces information loss in cross-layer feature interactions, which is especially critical for small and medium-sized models.` It may be that we have problems in expression, here `small and medium-sized models` refer to the N/S size model we proposed. In the case of SPAFPN-C2f-N/S, they do improve performance significantly (3.9\%/2.3\% mAP gains).



*[1] Chengcheng Wang, Wei He, Ying Nie, Jianyuan Guo, Chuanjian Liu, Yunhe Wang, and Kai Han. Gold-yolo: Efficient object detector via gather-and-distribute mechanism. Advances in Neural Information Processing Systems, 36, 2024.*

*[2] Chuyi Li, Lulu Li, Yifei Geng, Hongliang Jiang, Meng Cheng, Bo Zhang, Zaidan Ke, Xiaoming Xu, and Xiangxiang Chu. Yolov6 v3.0: A full-scale reloading. arXiv preprint arXiv:2301.05586, 2023.*

*[3] Jocher Glenn. Ultralytics yolov8 release v8.2.0. https://github.com/ultralytics/ultralytics/releases/tag/v8.2.0, 2023.*

*[4] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. Yolov9: Learning what you want to learn using programmable gradient information. arXiv preprint arXiv:2402.13616, 2024.*
