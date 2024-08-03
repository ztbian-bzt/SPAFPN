We express our sincere gratitude for your professional and comprehensive review. And we will provide responses to each of your quesion in the 'Weaknesses' section, 'Questions' section and 'Limitations' section.

- ***The main contribution of this paper needs further clarification. The authors identified the inefficient cross-scale propagation problem in PAFPN. However, the efficiency improvements in SPAFPN appear to derive from minor technical adjustments, such as the parametric sampling technique and the reduction of processed channels in the main feature extraction block. Notably, the primary contribution, "PFusion," seems to increase the delay in back-propagation (Line 145).***

**Reply:** First, we briefly summarize the main contributions of our work.
1. We adopt the strategy of "Light Fusion, Heavy Decouple" and propose a general neck architecture called Scarf PAFPN (SPAFPN). SPAFPN consists of a simple module called Pyramid Fusion, which is responsible for multi-scale feature fusion, and a Multi-Concat module, which is responsible for decoupling the fused features back to each scale.
2. MS COCO dataset is used for verification. In the case of only replacing the neck with SPAFPN, it has a significant improvement over the baseline model.
3. We changed a variety of backbones and achieved good detection results on MS COCO, which verified the generalization and portability of SPAFPN structure

We do identify the inefficient cross-scale propagation problem in PAFPN. And the 'inefficient' here does not refer to the speed of the model, but rather to the fact that the propagation of information between non-adjacent scales must rely on intermediate layers which makes propagation inefficient.

The parametric sampling technique can indeed improve the inter-layer propagation efficiency on the basis of SPAFPN structure. However, this is a discussion of additional performance improvements based on the SPAFPN structure. In Table 11 in Appendix B.3.1, we see that bilinear and Conv, the most popular down-sampling methods, still achieve good results (mAP 40.9%) for model of size N. Therefore, we only discuss the up-and-down sampling method in one paragraph (Lines 150-160). At the same time, this can also be considered as our thinking on how to better extract multi-scale features under "Light Fusion".

The reduction of processed channels in the main feature extraction block is a common technique in the real-time domain including real-time object detection. It utilizes duplicate gradient information within network optimization to reduce the number of parameters, FLOPs and improve the speed of the model. We will discuss this more specifically in our reply to the next weakness.

The paragraph containing Line 145 is as follows:
```As the front and back of this module are straight-through to the features of the three scales, the operations adopted on PFusion tend to spend more delay in back-propagation than those with the same amount of parameters and floating-point operations at other positions. Therefore, due to our pursuit of lightweight and easy training, unlike other multi-scale fusion studies, we do not add a feature extraction module with a complex structure to this fusion module and take the strategy of "Light Fusion, Heavy Decouple".``` 

Maybe we have a problem with our expression here.
What we are really saying here is that if a more complex operation (e.g., additional extraction block) is added after PFusion, it will result in higher latency than if it is added at other places such as after decoupling. This is the main reason why we adopted "Light Fusion". We can find from Table 6 in Section 4.3.2 that adding more complex operations after PFusion leads to a significant rise in train FPS. This is intolerable for us with insufficient computing resources. So, this delay only appears in ablation experiments.


- ***Some claims about SPAFPN lack clear illustration and experimental validation. For example, although the authors suggest processing only half the channels to reduce computational costs (Lines 196-200), they do not evaluate the effects of varying the proportion of unprocessed channels in their experiments.***

**Reply:** Processing only half the channels to reduce computational costs is a technique commonly used in the real-time domain. This method was first proposed in CSPNet [1], which utilizes duplicate gradient information in the feature extraction module during the network optimization process. The models I mentioned in the first paragraph of the Introduction that were published after 2020 more or less take advantage of this trick as well.

Conventionally, we refer to the processed channels as hidden channels. In [1], they also performed ablation experiments for the proportion of hidden channels in their Table 1. At your request, we have also done similar ablation experiments for our proposed model as follows:

**Ablation study on the proportion of hidden channels in CSP-DCN**
| Model |e| Param. | GFLOPs | mAP<sup>val</sup><sub>50-95</sub> | mAP<sup>val</sup><sub>50</sub> |
|---|---|---|---|---|---|
|SPAFPN-C2f-n|0.25|3.2M|9.7|39.6%|55.6%|
|SPAFPN-C2f-n|0.5|3.7M|10.7|41.2%|57.3%|
|SPAFPN-C2f-n|0.75|4.4M|11.9|41.3%|57.4%|


where, e represents the proportion of hidden channels. From the above experiments, it can be seen that when e=0.25, the performance will degrade quite significantly. However, when e=0.75, although the computational complexity increases, the performance of the model does not improve. Objectively, both e=0.25 and e=0.5 have some advantages, but in terms of performance, choosing e=0.5 is the better choice.



- ***I appreciate the efforts in this article. It might be helpful to revisit some of the concepts and paragraphs to ensure a more logical and rigorous paper. For example:***
- ***The section on object detection in the first paragraph (Lines 19-23) appears to be loosely related to the paper's main topic. The authors could consider removing this part and instead adding more details to discuss the background, technical developments, and challenges in real-time object detection.***

**Reply:**

- ***See Questions for more details.***
- ***The idea of "modularization" only mentioned twice in this paper and is not well explained. Is it the same as the concept of "Light Fusion, Heavy Decouple"? If true, the authors can consider avoiding the duplicated concept with different words.***

**Reply:** The idea of "modularization" here means versatility and portability. We can easily apply it to networks with PAFPN or similar structures. The feature extraction modules and sampling tools are also very easy to replace at the code level. Just as we can apply it to RT-DETR in a very short time.

- ***In Table 1, which result can indicate "a high level of speed" (as claimed in Line 235-236), and what does "it" (Line 235) refer to?***

**Reply:** Here is the imperfection of our expression. The sentence in Line 235-236 is `Also, despite the sacrifice in FPS, it still maintains a high level of speed in the above real-time object detection methods.` This sentence can be reorganized as `Also, despite sacrificing FPS compared to YOLOv8-M (464), SPAFPN-C2f-M (403) still maintains a high level of speed in the above real-time object detection methods (e.g YOLOv7(357), DAMO-YOLO-M(222), Gold-YOLO-M(351)).`


 - ***Some writing format issues: e.g., there should be a blank space before the parentheses, Pyramid Fusion(PFusion) -> Pyramid Fusion (PFusion).***

**Reply:** Thank you for pointing out our writing format issues. We will correct it in the next version of the paper and double-check for other writing format issues.
