We express our sincere gratitude for your professional and comprehensive review. And we will provide responses to each of your questions in Section 'Weaknesses'.

- ***The proposed method has a low gain (0.4) on large models, which means that spafpn is not necessary when the backbone features are strong enough.***

**Reply:** Thank you for your insightful comment regarding the gains on large models. 

It is a common phenomenon for real-time object detection to achieve small gains for common datasets (e.g., COCO) on large models, especially when improving only one of the backbone, neck, and head items of real-time object detection. For example, the Gold-YOLO [1] series uses YOLOv6 3.0 [2] as its baseline model, and its gains in N/S/M size are 2.6\%/1.1\%/0.7\% mAP, respectively. Compared to SOTA YOLOv8 [3] at that time, its gain was only 2.3\%/0.5\%/-0.4\% mAP. Moreover, in February this year, the open-source YOLOv9 series [4] had a gain of only 1.8\%/0.9\% mAP in S/M size compared with YOLOv8. This is the final gain of YOLOv9 after improving the backbone, neck, and head for real-time object detection.

Our method has a gain of 3.9\%/2.3\%/0.4\% mAP in N/S/M size over YOLOv8 when only changing the neck part and can achieve a gain of 5.2\%/3.2\%/1.1\% mAP when also changing the backbone part.

- ***The author claims that cross-scale interactions can help detect small and medium objects(L121-122), but this is not supported by experiments.***

**Reply:** Thank you for pointing out our mistake in expression. The sentence in L121-122 is `This reduces information loss in cross-layer feature interactions, which is especially critical for small and medium-sized models.` Here `small and medium-sized models` refer to the N/S size models we proposed. In the case of SPAFPN-C2f-N/S, they do improve performance significantly (3.9\%/2.3\% mAP gains). We will change this sentence to `This reduces information loss in cross-layer feature interactions, which is especially critical for the N/S size models' to avoid ambiguity in the next version of the paper.

- ***In the method section, the author mainly introduces the details of module design and stacks these modules to gain gains. There is a lack of explanation and insight into the module design.***

**Reply:** Thank you for pointing out the gaps in our writing. We will supplement our ideas and explanations for module design. The following explanations (`(1) Pyramid Fusion Module`, `(2) Multi-Concat Module` and `(3) CSP-DCN Module`) will be added to the next version of the paper in Section 3.2, Section 3.3 and Section 3.4, respectively.

**(1) Pyramid Fusion Module**

We design Pyramid Fusion Module to extract global fusion features from multiple scales. Many previous works (e.g. TopFormer[5], FAN[6], Gold-YOLO[1], CFP[7]) employ quite complex feature extraction modules in the multi-scale feature extraction process. However, according to the experimental results in Table 6, we find that although adopting complex feature extraction in the fusion module can bring better performance and inference FPS, it will significantly reduce the training FPS. This is obviously not friendly to researchers who are not rich in computing resources.

Furthermore, take the feature transfer from P5-1 to P3-2 as an example (Figure. 1).


$$P3-2 = β_3α_3(P3-1, β_4α_4(P4-1, P5-1))$$①

$$P3-2 = β_3α_3(F(P3-1, P4-1, \textcolor{#FF0000}{P5-1}), P3-1, β_4α_4(P4-1, P5-2, F(P3-1, P4-1,P5-1)))$$②

Where, $α_i$ is a simple alignment operation on the Pi scale, which generally only includes 1*1 convolution for adjusting channel number, up-and-down sampling, and Concatenate operations; $β_i$ is a feature extraction module on Pi scale, which has a much more complex structure than $α_i$; $F$ is the multi-scale feature fusion module.  Equation 1 is the usual PAFPN, and Equation 2 is the SPAFPN version. It can be found that if $F$ is constructed by simple operations similar to $β_i$, there is a P5-1 term in Formula 2 that can reach P3-2 by only one feature extraction module compared with Formula 1. This makes the features on P5-1 more directly accessible to P3-2.

Based on the above ablation experiments and conjectures, we propose the strategy of "Light Fusion, Heavy Decouple" to design Pyramid Fusion more simply. Since the Light Fusion strategy has been taken, to make up for the lack of fusion effect as much as possible, this motivates us to look at the sampling modules. Thus, the commonly used bilinear up-sampling and Conv down-sampling are replaced, which slightly improves the performance(refer to Table 11).

**(2) Multi-Concat Module**

In the structure of SPAFPN, due to the additional global input obtained by multi-scale fusion, how to decouple the fused global features and combine them with the features of the local layer is a problem that needs to be considered more.

As shown in Figure.3(c), we adopt the practice of concatenate channel after sampling in traditional FPN for adjacent scale input x_high/low and local scale input x_local. We refer to Squeeze-and-excitation[8] and Coordinate attention[9], and treat the global features x_global after aligning scales and channels as the corresponding weight elements of the local features, which are directly weighted by multiplication. Finally, the local features weighted by the global features are obtained. We also add a set of identity maps such that Mult-Concat can degenerate to the traditional Concat module when the input global feature is empty.

In addition, we propose the full-node strategy according to the characteristics of SPAFPN. In the BiFPN paper, the authors presented the idea that if there is only one input to a node and no feature fusion, this node has little impact on the feature network and can be removed. This view is reflected in practice in PAFPN-like structures, which neglected the nodes of layer P5 in FPN and layer P3 in PAN. These two nodes are consistent with a single input and no feature fusion. But in SPAFPN, the global features from PFusion can be extra inputs to these two edge nodes, making it possible for all nodes to be exploited(refer to Table 3).

**(3) CSP-DCN Module**

The CSP-DCN module is a neck module proposed to balance model performance and computational complexity. We learn from the structure of CSP[10] and adopt deformable convolution[11] to improve the understanding ability of the model while reducing the number of channels in the feature extraction module. The reduction in the number of channels is based on the presence of duplicate gradient information in the feature extraction module during the network optimization process (evidenced in [10]).


*[1] Chengcheng Wang, Wei He, Ying Nie, Jianyuan Guo, Chuanjian Liu, Yunhe Wang, and Kai Han. Gold-yolo: Efficient object detector via gather-and-distribute mechanism. Advances in Neural Information Processing Systems, 36, 2024.*

*[2] Chuyi Li, Lulu Li, Yifei Geng, Hongliang Jiang, Meng Cheng, Bo Zhang, Zaidan Ke, Xiaoming Xu, and Xiangxiang Chu. Yolov6 v3.0: A full-scale reloading. arXiv preprint arXiv:2301.05586, 2023.*

*[3] Jocher Glenn. Ultralytics yolov8 release v8.2.0. https://github.com/ultralytics/ultralytics/releases/tag/v8.2.0, 2023.*

*[4] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. Yolov9: Learning what you want to learn using programmable gradient information. arXiv preprint arXiv:2402.13616, 2024.*

*[5] Wenqiang Zhang, Zilong Huang, Guozhong Luo, Tao Chen, Xinggang Wang, Wenyu Liu, Gang Yu, and Chunhua Shen. Topformer: Token pyramid transformer for mobile semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12083–12093, 2022.*

*[6] Xuan Zhou and Xuefeng Wei. Feature aggregation network for building extraction from high-resolution remote sensing images. In Pacific Rim International Conference on Artificial Intelligence, pages 105–116. Springer, 2023.*

*[7] Yu Quan, Dong Zhang, Liyan Zhang, and Jinhui Tang. Centralized feature pyramid for object detection. IEEE Transactions on Image Processing, 2023.*

*[8] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7132–7141, 2018.*

*[9] Qibin Hou, Daquan Zhou, and Jiashi Feng. Coordinate attention for efficient mobile network design. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13713–13722, 2021.*

*[10] Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. Cspnet: A new backbone that can enhance learning capability of cnn. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 390–391, 2020.*

*[11] Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, et al. Internimage: Exploring large-scale vision foundation models with deformable convolutions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14408–14419, 2023.*
