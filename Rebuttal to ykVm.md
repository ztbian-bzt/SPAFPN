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



- ***I appreciate the efforts in this article. It might be helpful to revisit some of the concepts and paragraphs to ensure a more logical and rigorous paper. For example:***
- ***The section on object detection in the first paragraph (Lines 19-23) appears to be loosely related to the paper's main topic. The authors could consider removing this part and instead adding more details to discuss the background, technical developments, and challenges in real-time object detection.***

**Reply:**
 
