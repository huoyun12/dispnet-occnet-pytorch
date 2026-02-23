# Unsupervised Light Field Depth Estimation via Multi-view Feature Matching with Occlusion Prediction

Shansi Zhang, Graduate Student Member, IEEE, Nan Meng, Member, IEEE, and Edmund Y. Lam, Fellow, IEEE 

Abstract—Depth estimation from light field (LF) images is a fundamental step for numerous applications. Recently, learningbased methods have achieved higher accuracy and efficiency than the traditional methods. However, it is costly to obtain sufficient depth labels for supervised training. In this paper, we propose an unsupervised framework to estimate depth from LF images. First, we design a disparity estimation network (DispNet) with a coarseto-fine structure to predict disparity maps from different view combinations. It explicitly performs multi-view feature matching to learn the correspondences effectively. As occlusions may cause the violation of photo-consistency, we introduce an occlusion prediction network (OccNet) to predict the occlusion maps, which are used as the element-wise weights of photometric loss to solve the occlusion issue and assist the disparity learning. With the disparity maps estimated by multiple input combinations, we then propose a disparity fusion strategy based on the estimated errors with effective occlusion handling to obtain the final disparity map with higher accuracy. Experimental results demonstrate that our method achieves superior performance on both the dense and sparse LF images, and also shows better robustness and generalization on the real-world LF images compared to the other methods. 

Index Terms—Light field, unsupervised depth estimation, feature matching, occlusion prediction. 

# I. INTRODUCTION

A light field (LF) camera can capture both the intensities and directions of the light rays [1], [2] to obtain LF images, each of which consists of an array of sub-aperture images (SAIs) to record the scene from multiple viewpoints [3], [4]. As the LF images contain rich geometric information, useful clues are available for depth (disparity) estimation. Usually, depth estimation is an essential task for scene understanding and also a crucial step for various LF applications and researches, such as auto refocusing [5], scene reconstruction [6], novel view synthesis [7], [8], compressed sensing [9], and semantic segmentation [10]. 

Traditional methods for LF depth estimation mainly adopt two approaches. The first approach is to explore the structure of epipolar-plane images (EPIs) [11]–[14], where the pixels corresponding to the same scene point in different views form a line with a slope proportional to the disparity value. 

The work is supported in part by the Research Grants Council of Hong Kong (GRF 17201822) and by ACCESS — AI Chip Center for Emerging Smart Systems, Hong Kong SAR. (Corresponding author: Edmund Y. Lam) Shansi Zhang and Edmund Y. Lam are with the Department of Electrical and Electronic Engineering, The University of Hong Kong, Hong Kong SAR, China (e-mail: sszhang@eee.hku.hk; elam@eee.hku.hk). Nan Meng is with the Li Ka Shing Faculty of Medicine, The University of Hong Kong, Hong Kong SAR, China (e-mail: nanmeng@hku.hk). 

Another approach leverages the classical stereo matching to find the corresponding pixels among different views [15]– [18]. However, EPI-based methods are mainly applicable to the densely sampled LF images, and the traditional stereo matching-based methods usually suffer from heavy computational costs. Recently, many learning-based methods [19]–[24] have been proposed for LF depth estimation with improved accuracy and efficiency. They use a deep neural network to represent the estimator, which learns from the depth labels. However, it is costly to acquire sufficient, accurate depth annotations, especially for the real-world LF images, and the lack of training data usually leads to limited generalization ability. To alleviate the reliance on a large number of labeled data, some unsupervised learning-based methods [25]–[29] are developed, which implicitly learn the correspondences by a plain network with the photo-consistency constraint [30] to minimize the warping errors. However, these methods are mainly applicable to the dense LF images but are not effective enough for the sparse LF images with large disparity values. 

Our target is to develop an unsupervised LF depth framework applicable to both the dense and sparse LF images. We first design a disparity estimation network (DispNet) that explicitly performs multi-view feature matching by constructing memory-efficient cost volumes to learn the correspondences among the input views with a variety of disparity ranges. Our DispNet leverages a coarse-to-fine structure, with a coarse branch to estimate a initial disparity map and a refinement branch to further improve the disparity accuracy. Moreover, occlusion is a challenging issue in many LF tasks [8], [31], [32], and it leads to the violation of photo-consistency in LF depth estimation. To tackle this issue, we introduce an occlusion prediction network (OccNet) during training to predict the occlusion maps, which are used as the element-wise weights of the photometric loss to mitigate the effect of occlusions on the disparity learning. In order to fully utilize the views of each LF image, multiple view combinations are input to the DispNet to obtain multiple estimated disparity maps. We then propose a disparity fusion strategy with effective occlusion handling to obtain the final disparity map with higher accuracy. The main contributions of our work are summarized as follows: 

• We develop a DispNet, which employs a coarse-to-fine structure and performs multi-view feature matching to estimate disparity maps from different input combinations. 

• To tackle the occlusion issue, we introduce an OccNet for occlusion prediction, which aims to eliminate the adverse 

impact of occlusion on the training of DispNet. 

• With the multiple estimated disparity maps, we propose a disparity fusion strategy with occlusion handling to obtain the final disparity map. 

• Experimental results demonstrate that our method achieves superior performance on both the dense and sparse LF images with better robustness and generalization compared to the other methods. 

# II. RELATED WORK

The existing methods for LF depth estimation are reviewed in terms of the traditional methods and the learning-based methods. 

# A. Traditional Methods

Traditional methods for LF depth estimation mainly focus on the EPI structure and stereo matching. For the EPI-based methods, Wanner and Goldluecke [11] estimated the disparity maps locally by using EPI analysis, which works fast without the expensive matching cost minimization. Zhang et al. [12] proposed a spinning parallelogram operator (SPO) to locate the lines in EPI and calculate their slopes to acquire the depth information. Zhang et al. [13] exploited the line structure of EPI and the locally linear embedding (LLE) to estimate the local depth by minimizing the matching cost. Sheng et al. [14] developed a strategy to extract EPIs in multiple directions besides the horizontal and vertical EPIs to calculate the local depth, which is combined with the predicted occlusion boundaries to obtain the final depth map. 

For the stereo matching-based methods, Jeon et al. [15] constructed a cost volume to estimate the multi-view stereo correspondences, which was used to optimize the depths in weak texture regions. Then, the local depth map was refined iteratively by fitting the local quadratic function. Tao et al. [16] leveraged the shading information to improve the local shape estimation from defocus and correspondence, and developed a framework that exploits LF angular coherence for depth and shading optimization. Lee et al. [17] computed binary maps through foreground–background separation to obtain the disparity maps. Huang et al. [18] proposed a stereo matching algorithm using an empirical Bayesian framework, which employs the pseudo-random field to explore the statistical cues of LF. Zhang et al. [33] leveraged graph spectral analysis to exploit the angular and spatial structure information for depth estimation. 

Occlusion issue is often encountered in LF depth estimation since the photo-consistency assumption does not hold in the occlusion regions. There are some methods focusing on tackling this issue. Wang et al. [34] found that the photoconsistency still holds in about half of the views when the occlusions exist, and they predicted the occlusions, which was used as a regularizer to improve depth estimation. Williem et al. [35] focused on the robust depth estimation from noisy LF with occlusions. They introduced two data costs with angular entropy metric and adaptive defocus response to handle the occlusions and noises. Chen et al. [36] proposed a method with 

partially occluded region detection through super-pixel regularization and showed that even a simple least square model can achieve superior depth estimation after manipulating the label confidence and edge strength. 

These traditional methods usually involve complex optimization process and long execution time, and cannot achieve a good balance between the accuracy and efficiency. 

# B. Learning-based Methods

The recent work on LF depth estimation mainly focuses on the learning-based methods, which usually achieves higher accuracy and inference efficiency. Most existing learningbased methods adopt supervised training using depth labels. Sun et al. [19] developed a convolutional neural network (CNN) to estimate LF disparity by extracting enhanced EPI features. Heber et al. [20] proposed a U-shaped network to extract LF geometric information, with 3D convolutional layers to examine the EPI volumes for robust depth prediction. Shin et al. [21] developed a CNN framework by taking as input the views from different angular dimensions. They also proposed some data augmentation methods for LF to overcome the deficiency of training data. Shi et al. [22] proposed a framework to learn depth from the dense and sparse LF images with three steps, including initial depth estimation by a finetuned network, occlusion-aware depth fusion and refinement by an additional network. Tsai et al. [23] proposed a view selection network by learning an attention map to estimate the contribution of each view on depth. The attention map was constrained to be symmetric in accordance with the LF views. Chen et al. [24] developed a multi-level fusion network, which contains four branches to perform intra-branch and interbranch fusion, and incorporates attention to select the features that can provide more useful information for depth. Wang et al. [37] proposed a fast approach to construct matching cost for LF depth estimation, which does not require any shifting operation and can also handle the occlusions. These supervised methods heavily rely on the labeled data, which results in poor generalization ability when the labeled data are not sufficient. 

Unsupervised methods can overcome the reliance on the labeled data. Peng et al. [25] proposed an unsupervised CNN framework by designing a combined loss with compliance and divergence constraints to estimate LF disparity. Zhou et al. [26] developed an unsupervised monocular LF depth network, which was trained by the improved photometric losses and takes only one view as input. Jin et al. [27] proposed an unsupervised occlusion-aware framework by exploring the angular coherence among different LF subsets. Iwatsuki et al. [28] developed an unsupervised learning framework with pixel-wise weights to evaluate the warping errors and an edge loss to enforce edge alignment between the image and the disparity map. Lin et al. [29] proposed to integrate the traditional LF constraints into an unsupervised framework with an adaptive spatial-angular consistency loss. These methods directly output the disparity values from the last convolution layer without explicitly learning the correspondences among different views, which leads to poor performance when applied to the sparse LF images with large disparity values. 


(a) Input views


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/8d111139852dc617ebde8d0259314479256ded680bcce6218950fbddc190ca06.jpg)



(b) Disparity estimation for different input combinations


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/72eef8055c4a6218322ceac24569e2d6c06d4423eced0678bedd1cde9be3119d.jpg)



(c) Disparity fusion


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/239c24b17b353ae4830f689adeb42cb434444a805b77b958c949154b2d690e4d.jpg)



Fig. 1. (a) The views input to the DispNet and used for determining the errors during fusion. (b) The DispNet takes as inputs the central view $\mathbf { I } _ { c }$ , the left source view $\mathbf { I } _ { l }$ and the right source view ${ \mathbf { I } } _ { r }$ . The views from the same column are rotated by $9 0 °$ . Multiple disparity maps (after scaling and rotation) are obtained from different input combinations. (c) The estimated disparity maps are fused according to their estimated errors using the auxiliary views to obtain the final disparity map.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/eca03e0ce3534a0ab983ec547b8aa6f30937bc955e7e7ddad992d4060757bd50.jpg)



Fig. 2. The central view and the warping error maps of the left and right views. The occlusion regions for the left and right views usually near the object boundaries and locate in the opposite positions.


# III. PROPOSED METHOD

We first describe our overall framework for LF disparity estimation in Sec. III-A. Then, we introduce the architecture of our DispNet in Sec. III-B, the occlusion prediction method in Sec. III-C, and the loss functions for training in Sec. III-D. Finally, we introduce our disparity fusion strategy in Sec. III-E. 

# A. Overall Framework

A 4D LF image is represented as $\mathbf { L } \in \mathbb { R } ^ { U \times V \times X \times Y }$ , with angular index $( u , v )$ and spatial index $( x , y )$ . It consists of $U \times$ $V$ SAIs, each of which records the scene from one viewpoint with a spatial resolution of $X \times Y$ . Our target is to estimate the disparity map of the central view relative to its adjacent views. According to the LF geometry, the relationship between the central view and any other view in terms of the central disparity $\mathbf { d } ( x , y )$ is expressed as 

$$
\mathbf {L} \left(u _ {c}, v _ {c}, x, y\right) = \mathbf {L} (u, v, x + \mathbf {d} (x, y) \times \left(u _ {c} - u\right), \tag {1}
$$

$$
y + \mathbf {d} (x, y) \times (v _ {c} - v)),
$$

where $( u _ { c } , v _ { c } )$ is the angular position of the central view. 

The overall framework of our method is depicted in Fig. 1. Three views, including the central view $\mathbf { I } _ { c }$ , the left source view $\mathbf { I } _ { l }$ and the right source ${ \mathbf I } _ { r }$ view (“source” means that 

they are warped according to the disparity to reconstruct the central view), are fed to the DispNet. They are in the same row or column, and the two source views framed by the same color are located symmetrically to the central view. This input strategy can avoid the matching ambiguity caused by the occlusion without incorporating any redundant inputs, since each scene point is usually visible in at least two views of the three input views. Fig. 2 gives an intuitive illustration. The left view and right view are warped to the central view using the ground-truth disparity map of the central view, and their warping error maps are presented. The bright regions with large errors in the warping error maps correspond to the occlusion regions, which are visible in the central view but invisible in the left or right views. It can be seen that the occlusion regions for the left and right views usually near the object boundaries and locate in the opposite positions, which indicates that the pixel in the central view has correspondence in at least one view of the left and right views to enable accurate disparity estimation. 

For a $7 \times 7$ LF image (Fig. 1(a)), there are totally 6 input combinations. The views from the same column should be rotated by $9 0 °$ (counterclockwise) before being input to the network in order to convert the vertical disparity to be horizontal, and the corresponding output disparity map needs to be rotated by $- 9 0 ^ { \circ }$ (clockwise) to recover the orientation. In this way, only horizontal disparity estimation is involved to make the learning easier. Multiple disparity maps can be obtained from different input combinations by the shared DispNet, and they need to be scaled by the distance between the source views and the central view, as shown in Fig. 1(b). Using nonadjacent views to estimate disparity helps to alleviate the inaccurate estimation caused by the narrow LF baseline. Then, these disparity maps are fused based on their estimated errors using the auxiliary views (in the diagonal direction) to obtain the final disparity map (Fig. 1(c)). In what follows, we will introduce each part in detail. 


(a) Architecture of DispNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/dde07490f7423543102b1d9665f971fccf91fc40819e6db60de7b79d2fb44325.jpg)



(b) Feature Extractor


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/2f71578af8bfed04a4566c12086ec5ec6176fd4efa5502065aa11830757bfd75.jpg)



(c) Cost Filter


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/1f5a709d6aeceda2d6e1301153186f3ad19d43035116cb3386f33d0c10d49fb6.jpg)



(d) Disparity Regression


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/6e74760622769804b50379dc2c88efd7caa7cc8ae72be8d045ba9c3e601e38ea.jpg)



Fig. 3. (a) Architecture of DispNet. It consists of two branches with shared feature extractor and cost filters to estimate the coarse disparity map and residual map by constructing the coarse and residual cost volumes with variance-based feature matching. (b) Feature extractor. (c) Cost filter. (d) Disparity regression.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/f005ed13a3890b30c05615d48f5eb8c9084039d946a3d84680ae66d24b9a2966.jpg)



Fig. 4. Variance-based feature matching. The left and right source features are warped according to a disparity sample $s _ { i }$ to match the reference feature. The element-wise variance of the warped features and reference feature is calculated to obtain the matching cost at the disparity sample $s _ { i }$ . The final cost volume is obtained by concatenating the matching costs at all the disparity samples.


# B. Disparity Estimation

The architecture of our DispNet is shown in Fig. 3. The central view $\mathbf { I } _ { c }$ and the two source views $\mathbf { I } _ { l }$ and ${ \mathbf I } _ { r }$ are fed to a shared feature extractor. The feature extractor (Fig. 3(b)) 

consists of several residual blocks, each of which has two $3 \times 3$ convolution layers with leaky Rectified Linear Unit (ReLU) activation, and an atrous spatial pyramid pooling (ASPP) [38] block to encode multi-scale features, which contains three dilated convolutions with rates 3, 6 and 8 and a global average pooling (GAP) operation for global receptive field. 

The extracted features of the three views are used to construct the cost volumes through variance-based feature matching. The detailed procedures of constructing the coarse cost volume is illustrated in Fig. 4. First, we prescribe $D$ equally spaced disparity samples with the minimum value $s _ { \mathrm { m i n } }$ and the maximum value $s _ { \mathrm { m a x } }$ , expressed by a vector $\mathbf { s } = [ s _ { \mathrm { m i n } } , \cdot \cdot \cdot , s _ { i } , \cdot \cdot \cdot , s _ { \mathrm { m a x } } ] ^ { T }$ . Given the features of the three views with a size $C \times X \times Y$ $C$ is the channel number), the left and right source features are warped to match the reference (central) feature by a disparity sample $s _ { i }$ from s. Then, the element-wise variance of the warped features and reference feature is calculated to measure the difference, which is treated as the matching cost at disparity sample $s _ { i }$ (accurate disparity at a pixel should lead to a small variance). The cost volume is obtained by concatenating the matching costs at all the disparity samples, with a size $C \times D \times X \times Y$ . Compared to the common approach used in stereo matching [39], which builds the cost volume by concatenating the reference feature and warped source features, our variance-based feature matching is much more memory-efficient and can also adapt to any number of input views without increasing the size of cost volume. 

The coarse cost volume is further processed by the cost filters (Fig. 3(c)), each of which consists of several 3D residual blocks with skip connections. The blocks with stride $= 2$ aim to downsample the cost volume to reduce memory consumption. Then, a coarse disparity regression module (Fig. 3(d)) is employed, with 3D convolution layers to yield the coarse 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/db3e8b9805a0988fdca4d40d2802751ee817e9374c89548540ba35b9fc2df8e4.jpg)



Fig. 5. Training with predicted occlusion maps. The estimated disparity map from the DispNet is used to warp the two source views to the central view. Then, the concatenation of the warped views and the disparity map is input to the OccNet to predict the confidence maps for reconstructing the central view, which are treated as occlusion maps and used as the pixel-wise weights of the photometric loss $\ell _ { \mathrm { w p m } }$ . The OccNet is trained by the reconstruction loss $\ell _ { \mathrm { r e c } }$ . The arrows for DispNet flow, warping, OccNet flow and loss are distinguished by different colors.


cost $\mathbf { c } _ { \mathrm { c o a } } \in \mathbb { R } ^ { D \times X \times Y }$ , and the coarse disparity map $\tilde { \mathbf { d } } _ { \mathrm { { c o a } } }$ is obtained by 

$$
\tilde {\mathbf {d}} _ {\text {c o a}} (x, y) = \mathbf {s} \cdot \operatorname {s o f t m a x} \left(\mathbf {c} _ {\text {c o a}} (x, y)\right), \tag {2}
$$

where softmax $( \cdot )$ is the softmax function to obtain the probability of each disparity sample, and · denotes the inner product. 

To further refine the disparity map, a residual cost volume is constructed by using the same extracted features. Similarly, we set a residual sample vector $\mathbf { s } ^ { \prime } = [ s _ { \mathrm { m i n } } ^ { \prime } , \cdot \cdot \cdot , s _ { i } ^ { \prime } , \cdot \cdot \cdot , s _ { \mathrm { m a x } } ^ { \prime } ] ^ { T }$ , · · · , s′max] with the minimum value $s _ { \mathrm { m i n } } ^ { \prime }$ and the maximum value $s _ { \mathrm { m a x } } ^ { \prime }$ . The left and right source features are warped according to the coarse disparity map plus a residual sample $s _ { i } ^ { \prime }$ from $\mathbf { s } ^ { \prime }$ , with $\mathbf { \widetilde { d } } _ { \mathrm { c o a } } ( x , y ) + \mathbf { \widetilde { s } } _ { i } ^ { \prime }$ at each position, to match the reference feature. The value and interval of residual sampling are much smaller than that of coarse sampling in order to improve the disparity accuracy. The matching cost at each residual sample is derived by calculating the variance of the warped and reference features, and the residual cost volume is obtained by concatenating the matching costs at all the residual samples. Then, the residual cost volume is processed by the shared cost filters and the residual disparity regression module to derive the residual map $\tilde { \mathbf { d } } _ { \mathrm { r e s } }$ , with 

$$
\tilde {\mathbf {d}} _ {\text {r e s}} (x, y) = \mathbf {s} ^ {\prime} \cdot \operatorname {s o f t m a x} \left(\mathbf {c} _ {\text {r e s}} (x, y)\right), \tag {3}
$$

where $\mathbf { c } _ { \mathrm { { r e s } } } \in \mathbb { R } ^ { D \times X \times Y }$ is the residual cost. 

Thus, the refined disparity map $\tilde { \mathbf { d } }$ is obtained by 

$$
\tilde {\mathbf {d}} = \tilde {\mathbf {d}} _ {\mathrm {c o a}} + \tilde {\mathbf {d}} _ {\mathrm {r e s}}. \tag {4}
$$

When the input views are not adjacent, the output disparity maps from DispNet need to be scaled according to the view distance. In addition, the disparity maps predicted by the views from the same column need to be rotated by $- 9 0 ^ { \circ }$ to recover the orientation. Thus, the estimated central disparity map $\hat { \mathbf { d } }$ is obtained by 

$$
\hat {\mathbf {d}} = \left\{ \begin{array}{l l} \frac {\tilde {\mathrm {d}}}{u _ {c} - u _ {l}}, & \text {f r o m t h e s a m e r o w} \\ \operatorname {r o t} _ {- 9 0 ^ {\circ}} \left(\frac {\tilde {\mathrm {d}}}{v _ {c} - v _ {l}}\right), & \text {f r o m t h e s a m e c o l u m n} \end{array} \right. \tag {5}
$$

where $u _ { l }$ and $v _ { l }$ are the angular coordinates of the left source view, and rot−90◦ means rotating by $- 9 0 ^ { \circ }$ . 

# C. Occlusion Prediction

During training, $\tilde { \mathbf { d } }$ is used to warp the left and right source views to the central view, yielding $\mathbf { I } _ { l  c }$ and $\mathbf { I } _ { r  c }$ , with 

$$
\mathbf {I} _ {l \rightarrow c} (x, y) = \mathbf {I} _ {l} \left(x + \tilde {\mathbf {d}} (x, y), y\right), \tag {6}
$$

$$
\mathbf {I} _ {r \rightarrow c} (x, y) = \mathbf {I} _ {r} \big (x - \tilde {\mathbf {d}} (x, y), y \big). \tag {7}
$$

To predict the occlusion regions for the two source views, we introduce an OccNet that takes the concatenation of $\mathbf { I } _ { l  c }$ , $\mathbf { I } _ { r  c }$ and $\tilde { \mathbf { d } }$ as inputs, as shown in Fig. 5. The OccNet adopts a U-shape structure with residual blocks and skip connections. A softmax activation is used in the last convolution layer to output the confidence maps, $\mathbf { O } _ { l }$ and ${ \bf O } _ { r }$ for $\mathbf { I } _ { l  c }$ and $\mathbf { I } _ { r  c }$ respectively, which are used to reconstruct the central view, with, 

$$
\mathbf {I} _ {\text {r e c}} = \mathbf {O} _ {l} \odot \mathbf {I} _ {l \rightarrow c} + \mathbf {O} _ {r} \odot \mathbf {I} _ {r \rightarrow c}, \tag {8}
$$

where $\mathbf { I } _ { \mathrm { r e c } }$ is the reconstructed central view, $\odot$ represents the element-wise multiplication, and $\mathbf { O } _ { l } ( x , y ) \substack { + } \mathbf { O } _ { r } ( x , y ) = 1$ . The OccNet is trained by the reconstruction loss, with 

$$
\ell_ {\text {r e c}} = \left\| \mathbf {I} _ {\text {r e c}} - \mathbf {I} _ {c} \right\| _ {1}, \tag {9}
$$

where $\| \cdot \| _ { 1 }$ is the $\ell _ { 1 }$ -norm operator. 

The photometric loss for unsupervised disparity training is based on the photo-consistency assumption, which does not hold in the occlusion regions. Therefore, the pixel-wise weights of the photometric loss in the occlusion regions are expected to be small. The confidence maps $\mathbf { O } _ { l }$ and ${ \bf O } _ { r }$ can be treated as the occlusion maps of the two warped views, since 

the occlusion regions usually lead to relatively larger warping errors, and therefore less confidence for the reconstruction. Thus, we propose the weighted photometric loss with the occlusion maps, expressed as 

$$
\begin{array}{l} \ell_ {\mathrm {w p m}} = \frac {1}{X Y} \sum_ {x, y} \mathbf {O} _ {l} ^ {(x, y)} \odot | \mathbf {I} _ {l \rightarrow c} ^ {(x, y)} - \mathbf {I} _ {c} ^ {(x, y)} | \tag {10} \\ + \frac {1}{X Y} \sum_ {x, y} \mathbf {O} _ {r} ^ {(x, y)} \odot | \mathbf {I} _ {r \rightarrow c} ^ {(x, y)} - \mathbf {I} _ {c} ^ {(x, y)} |. \\ \end{array}
$$

The OccNet plays an important role during the training of DispNet. First, it helps to enforce the similarity between the warped source views and the central view for disparity learning through the reconstruction loss, as it is jointly trained with the DispNet. Second, with the improvement of estimated disparity map, large warping errors mainly lie in the occlusion regions, which can be identified by the OccNet to alleviate the adverse impact on further improvement. Note that the OccNet is only employed during training to address the occlusion issue and assist the disparity learning without influencing the inference efficiency. 

# D. Loss Function

In addition to the weighted photometric loss $\ell _ { \mathrm { w p m } }$ and the reconstruction loss $\ell _ { \mathrm { r e c } }$ , we apply a structural similarity (SSIM) loss [40] to further enforce the similarity, with 

$$
\ell_ {\text {S S I M}} = 1 - \frac {\operatorname {S S I M} \left(\mathbf {I} _ {l \rightarrow c} , \mathbf {I} _ {c}\right) + \operatorname {S S I M} \left(\mathbf {I} _ {r \rightarrow c} , \mathbf {I} _ {c}\right)}{2}. \tag {11}
$$

To improve the smoothness of the estimated disparity map while preserving the boundary structures of the objects, we leverage the structure-aware smoothness loss [41], [42], expressed as 

$$
\ell_ {\mathrm {s m d}} = \frac {1}{X Y} \sum_ {x, y} | \nabla \tilde {\mathbf {d}} ^ {(x, y)} | \odot \exp (- \eta | \nabla \mathbf {I} _ {c} ^ {(x, y)} |), \tag {12}
$$

where $\nabla$ denotes the gradients along both the horizontal and vertical directions, and $\eta$ is a hyperparameter for structure preservation. 

Moreover, we apply a similar smoothness loss to the occlusion map, with 

$$
\ell_ {\mathrm {s m o}} = \frac {1}{X Y} \sum_ {x, y} | \nabla \mathbf {O} _ {l} ^ {(x, y)} | \odot \exp (- \eta | \nabla \mathbf {I} _ {c} ^ {(x, y)} |), \tag {13}
$$

which is only applied to $\mathbf { O } _ { l }$ since $\mathbf { O } _ { r } = \mathbf { 1 } - \mathbf { O } _ { l }$ 

The DispNet and OccNet are trained simultaneously by the following full loss, 

$$
\ell_ {\text {f u l l}} = \ell_ {\mathrm {w p m}} + \ell_ {\mathrm {r e c}} + \alpha_ {1} \ell_ {\mathrm {S S I M}} + \alpha_ {2} \ell_ {\mathrm {s m d}} + \alpha_ {3} \ell_ {\mathrm {s m o}}, \tag {14}
$$

where $\ell _ { \mathrm { w p m } }$ and $\ell _ { \mathrm { r e c } }$ are the necessary losses, and the others are the auxiliary losses with coefficients $\alpha _ { 1 } \sim \alpha _ { 3 }$ . Note that these loss terms are also applied to the coarse disparity map $\tilde { \mathbf { d } } _ { \mathrm { { c o a } } }$ , but we omit the procedures for simplicity. 

# E. Multi-disparity Fusion Based on Estimated Errors

Multiple disparity maps can be obtained by the DispNet with different input combinations. To obtain the final disparity map, we propose a disparity fusion strategy to merge these disparity maps. 

Suppose that we have $n$ estimated disparity maps $\{ \hat { \mathbf { d } } _ { j } \} _ { j = 1 } ^ { n }$ and $Z$ auxiliary views to evaluate their accuracy. With each disparity map, the auxiliary views are warped to the central view, and the warping errors are calculated. However, some of the warping errors are not accurate due to the occlusions. If occlusions exist in some of the auxiliary views, the warping errors at the corresponding positions would be large, resulting in large variances among the $Z$ warping errors. Thus, we calculate the standard deviation of the warping errors to judge if occlusion exists at each position, and obtain a binary mask. Here, we define $\epsilon _ { j } ~ \in ~ \bar { \mathbb { R } ^ { X \times Y \times Z } }$ , indexed by $( x , y , z )$ , as the warping error maps of the $Z$ auxiliary views using the disparity map $\hat { \mathbf { d } } _ { j }$ , and $\sigma _ { z } ( \cdot )$ as the standard deviation along $Z$ dimension. Then, a binary mask $\mathbf { M } _ { j }$ is formulated as 

$$
\mathbf {M} _ {j} ^ {(x, y)} = \left\{ \begin{array}{l l} 1, & \sigma_ {z} \left(\boldsymbol {\epsilon} _ {j} ^ {(x, y, z)}\right) > \theta (q) \\ 0, & \text {o t h e r w i s e} \end{array} \right. \tag {15}
$$

where $\theta ( \cdot )$ is to determine the threshold for occlusion using the quantile $q$ of $\sigma _ { z } ( \epsilon _ { j } ^ { ( x , y , z ) } )$ . The effect of $q$ is analyzed in Sec. IV-D4. 

For a pixel with occlusion, we use the median value of the warping errors to represent its estimated error, which can eliminate the influence of large warping errors due to occlusion. Otherwise, the estimated error is represented by the mean of all the warping errors. Then, the estimated error map $\mathbf { e } _ { j }$ for $\hat { \mathbf { d } } _ { j }$ is obtained by 

$$
\mathbf {e} _ {j} ^ {(x, y)} = \operatorname {m e d i a n} _ {z} \left(\boldsymbol {\epsilon} _ {j} ^ {(x, y, z)}\right) \odot \mathbf {M} _ {j} + \operatorname {m e a n} _ {z} \left(\boldsymbol {\epsilon} _ {j} ^ {(x, y, z)}\right) \odot (\mathbf {1} - \mathbf {M} _ {j}), \tag {16}
$$

where $\mathrm { m e d i a n } _ { z } ( \cdot )$ and $\mathrm { m e a n } _ { z } ( \cdot )$ denote the median and mean along $Z$ dimension. 

With the above procedures, we can derive the error maps $\{ { \bf e } _ { j } \} _ { j = 1 } ^ { n }$ for all the disparity maps. Next, we need to merge these disparity maps according to their estimated errors. Here, we consider several different fusion approaches. The first one is the minimum error fusion by choosing the pixels with the minimum errors from each disparity map, and the final disparity map $\hat { \mathbf { d } } _ { \mathrm { f i n a l } }$ is derived by 

$$
j ^ {\prime} = \arg \min  _ {j} \left(\mathbf {e} _ {j} (x, y)\right), \tag {17}
$$

$$
\hat {\mathbf {d}} _ {\text {f i n a l}} (x, y) = \hat {\mathbf {d}} _ {j ^ {\prime}} (x, y). \tag {18}
$$

The second approach is the weighted fusion, where the weights are obtained by the softmax function and negatively correlated with the errors. Moreover, we can choose different numbers of disparity pixels for weighted fusion at each position. If $n ^ { \prime }$ $( n ^ { \prime } \leq n )$ ) disparity pixels with the smallest errors 

are used at each position, the final disparity map is obtained by 

$$
\mathbf {W} (x, y) = \operatorname {s o f t m a x} \left(- \mathbf {E} (x, y)\right), \tag {19}
$$

$$
\hat {\mathbf {d}} _ {\text {f i n a l}} (x, y) = \sum_ {j = 1} ^ {n ^ {\prime}} \mathbf {w} _ {j} (x, y) \times \hat {\mathbf {d}} _ {j} (x, y), \tag {20}
$$

where $\mathbf { E } ( x , y ) = \{ \mathbf { e } _ { j } ( x , y ) \} _ { j = 1 } ^ { n ^ { \prime } }$ denotes the smallest $n ^ { \prime }$ errors at position $( x , y )$ , and $\mathbf { W } ( x , y ) = \{ \mathbf { w } _ { j } ( x , y ) \} _ { j = 1 } ^ { n ^ { \prime } }$ denotes their corresponding weights. 

# IV. EXPERIMENTS

# A. Datasets

To evaluate the model performance comprehensively, we used both the synthetic and real-world LF datasets. The synthetic LF datasets include both the densely and sparsely sampled LF images. The synthetic dense LF images are from the HCI dataset [43] and the DLF dataset [22], with a disparity range $[ - 4 , 4 ]$ pixels between the adjacent views. The synthetic sparse LF images are from the SLF dataset [22], with a much larger disparity range $[ - 2 0 , 2 0 ]$ pixels. The real-world LF images are from the Stanford Lytro LF Archive [44], Kalantari [45] and EPFL LF [46] datasets. They can also be treated as the dense LFs since the disparity values between the adjacent views are very small. For all the LF images, the central $7 \times 7$ SAIs were used to estimate the disparity maps for the central views. 

# B. Implementation Details

Regarding the DispNet, the maximum channel number within the feature extractor is 128, and the number of cost filters is 2. For the dense LFs, The range of disparity samples was set to $[ - 1 2 , 1 2 ]$ (three times the disparity range $[ - 4 , 4 ] )$ ) with a coarse sampling interval 1, and the range of residual samples was set to $[ - 1 , 1 ]$ with a finer sampling interval 0.1. For the sparse LFs, only the central view and its adjacent views were used since a large disparity range would lead to high memory consumption. Thus, the range of disparity samples was set to $[ - 2 0 , 2 0 ]$ with a coarse sampling interval 1.2, and the range of residual samples was set to $[ - 2 , 2 ]$ with a finer sampling interval 0.12. For the OccNet, the maximum channel number is 64. Therefore, it is very lightweight with only 0.113 M parameters. 

During training, all the views were cropped to $2 5 6 \times 2 5 6$ randomly. The learning rate was set to $1 \times 1 0 ^ { - 3 }$ initially, and multiplied by 0.8 every 50 epochs. Moreover, we empirically set $\alpha _ { 1 } ~ = ~ 1$ for the SSIM loss, and $\eta ~ = ~ 1 0 0$ , $\alpha _ { 2 } ~ = ~ 0 . 1$ , $\alpha _ { 3 } = 0 . 0 5$ for the smoothness losses. The DispNet and OccNet were jointly trained using the loss in Eq. 14 with the Adam optimizer for about 500 epochs. 

During inference, multiple view combinations were input to the DispNet to obtain multiple disparity maps. For the dense LFs, the input combinations (expressed by the angular coordinates) are $[ ( u _ { c } - 3 , v _ { c } ) , ( u _ { c } , v _ { c } ) , ( u _ { c } + 3 , v _ { c } ) ]$ , $[ ( u _ { c } \mathrm { ~ - ~ }$ $2 , v _ { c } ) , ( u _ { c } , v _ { c } ) , ( u _ { c } + 2 , v _ { c } ) ]$ , $[ ( u _ { c } , v _ { c } - 3 ) , ( u _ { c } , v _ { c } ) , ( u _ { c } , v _ { c } + 3 ) ]$ , $[ ( u _ { c } , v _ { c } \mathrm { ~ - ~ } 2 ) , ( u _ { c } , v _ { c } ) , ( u _ { c } , v _ { c } \mathrm { ~ + ~ } 2 ) ]$ , and the weighted fusion with $n ^ { \prime } \ = \ 2$ was used to obtain the final disparity 

map. For the sparse LFs, the input combinations are $[ ( u _ { c } \mathrm { ~ - ~ }$ $1 , v _ { c } )$ , (uc, vc), $\left( u _ { c } + 1 , v _ { c } \right) ]$ , $[ ( u _ { c } , v _ { c } - 1 ) , ( u _ { c } , v _ { c } ) , ( u _ { c } , v _ { c } + 1 ) ]$ , and the minimum error fusion was used since there are only two estimated disparity maps. The quantile $q$ in Eq. 15 was set to 0.95. 

# C. Comparison

We compared our method with several state-of-the-art LF depth estimation methods, including both the supervised and unsupervised methods, on a variety of LF datasets. 

1) Evaluation on Synthetic Dense LF Images: First, we list the quantitative results of different methods on several scenes from the HCI dataset in Table I, including two supervised methods, EPINet [21] and LFAttNet [23], two non-learningbased methods, OCC [34] and FBS [17], and four unsupervised methods, UnCNN [25], UnMonocular [26], UnPlug [28], and UnOcc [27]. The evaluation metrics are the mean square error (MSE $\times 1 0 0 _ { , }$ and bad pixel ratios (BPR) [43] with thresholds 0.07, 0.03 and 0.01 (lower is better for all). The results of the other methods were obtained from the benchmark of the HCI dataset [43] or the original papers 1. It can be seen that our method still has some gaps with the supervised methods but can generally outperform the other unsupervised methods. Some of the visual results (Dino and Sideboard), with the error maps relative to the ground truths, are presented in Fig. 6, where we can find that our estimated disparity maps have better visual qualities with smaller errors than the other non-learning-based and unsupervised methods. 

Then, we compared the performance of different methods on several scenes from the DLF dataset [22], as recorded in Table II. The methods for comparison include EPINet [21], LFAttNet [23], UnCNN [25], UnMonocular [26], UnCon [29], and UnOcc [27]. We re-built these models and trained them using the same synthetic dense LF datasets since there are no off-the-shelf models for evaluation. From our experiments, we can see that our method obtains better quantitative results than the other unsupervised methods. Compared to the supervised methods, our method still has some gaps in terms of the MSE, but with comparable or lower BPRs. Fig. 7 presents the visual results on the scenes, Toys and Antiques, which reflects that our estimated disparity maps are more visually compelling with proper smoothness and clear object details. 

2) Evaluation on Synthetic Sparse LF Images: To evaluate the performance of different methods on the sparse LF images, we retrained them using the SLF dataset [22]. The quantitative results of several scenes in terms of the MSE and BPR with thresholds 0.3, 0.1 and 0.05 are listed in Table III. It can be seen that our method achieves comparable quantitative results with the supervised method LFAttNet, and has better results than the others. Both our method and LFAttNet predict the probability distribution of the disparity samples by constructing cost volumes, while the other methods directly output the disparity values from the last convolution layer, which increases the difficulty to learn large disparities for the networks. The visual results on the scenes, Lion and Rooster 

1The results of UnOcc [27] and UnPlug [28] are from their published papers, and they provide only the MSE and BPR with threshold 0.07. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/4e131ec333cf875a181c7cf07b6b9d85f06bab24664adbcf66bb3b25bcf9fb5c.jpg)



Fig. 6. Visual results of different methods on the scenes from the HCI dataset [43], with the error maps relative to the ground truths.



TABLE I EVALUATION ON THE SCENES FROM HCI DATASET. BOLD: BEST AMONG THE UNSUPERVISED METHODS.


<table><tr><td rowspan="2">Methods</td><td colspan="4">Dino</td><td colspan="4">Sideboard</td><td colspan="4">Backgammon</td><td colspan="4">Pyramids</td></tr><tr><td>MSE↓(×100)</td><td>BPR↓(0.07)</td><td>BPR↓(0.03)</td><td>BPR↓(0.01)</td><td>MSE↓(×100)</td><td>BPR↓(0.07)</td><td>BPR↓(0.03)</td><td>BPR↓(0.01)</td><td>MSE↓(×100)</td><td>BPR↓(0.07)</td><td>BPR↓(0.03)</td><td>BPR↓(0.01)</td><td>MSE↓(×100)</td><td>BPR↓(0.07))</td><td>BPR↓(0.03)</td><td>BPR↓(0.01)</td></tr><tr><td colspan="17">Supervised</td></tr><tr><td>EPINet [21]</td><td>0.167</td><td>1.286</td><td>3.452</td><td>22.401</td><td>0.742</td><td>4.277</td><td>10.824</td><td>37.999</td><td>3.629</td><td>3.580</td><td>6.289</td><td>20.899</td><td>0.008</td><td>0.192</td><td>0.913</td><td>11.876</td></tr><tr><td>LFAAttNet [23]</td><td>0.093</td><td>0.848</td><td>2.340</td><td>12.224</td><td>0.531</td><td>2.870</td><td>7.243</td><td>20.739</td><td>3.648</td><td>3.126</td><td>3.984</td><td>11.582</td><td>0.004</td><td>0.195</td><td>0.489</td><td>2.063</td></tr><tr><td colspan="17">Non-learning</td></tr><tr><td>OCC [34]</td><td>0.944</td><td>15.366</td><td>50.167</td><td>88.810</td><td>2.073</td><td>17.910</td><td>50.550</td><td>84.653</td><td>22.782</td><td>13.522</td><td>44.899</td><td>91.402</td><td>0.077</td><td>1.450</td><td>25.574</td><td>92.860</td></tr><tr><td>FBS [17]</td><td>0.664</td><td>8.427</td><td>23.533</td><td>65.390</td><td>1.072</td><td>13.296</td><td>32.516</td><td>70.042</td><td>5.805</td><td>10.162</td><td>22.181</td><td>65.407</td><td>0.029</td><td>0.549</td><td>5.705</td><td>78.243</td></tr><tr><td colspan="17">Unsupervised</td></tr><tr><td>UnCNN [25]</td><td>1.807</td><td>23.660</td><td>47.876</td><td>78.724</td><td>3.149</td><td>26.173</td><td>45.384</td><td>82.924</td><td>11.034</td><td>31.783</td><td>65.583</td><td>87.987</td><td>0.191</td><td>10.849</td><td>43.972</td><td>79.113</td></tr><tr><td>UnMonocular [26]</td><td>1.031</td><td>5.402</td><td>14.757</td><td>43.258</td><td>2.770</td><td>10.947</td><td>23.646</td><td>61.406</td><td>11.833</td><td>12.311</td><td>28.524</td><td>68.312</td><td>0.027</td><td>0.262</td><td>8.725</td><td>35.594</td></tr><tr><td>UnPlug [28]</td><td>0.788</td><td>6.178</td><td>-</td><td>-</td><td>1.999</td><td>12.766</td><td>-</td><td>-</td><td>9.399</td><td>14.200</td><td>-</td><td>-</td><td>0.022</td><td>0.658</td><td>-</td><td>-</td></tr><tr><td>UnOcc [27]</td><td>0.63</td><td>8.25</td><td>-</td><td>-</td><td>1.79</td><td>14.20</td><td>-</td><td>-</td><td>6.684</td><td>14.371</td><td>-</td><td>-</td><td>0.213</td><td>7.348</td><td>-</td><td>-</td></tr><tr><td>Ours</td><td>0.650</td><td>6.586</td><td>16.722</td><td>46.380</td><td>1.738</td><td>12.013</td><td>25.848</td><td>58.744</td><td>5.740</td><td>10.710</td><td>18.452</td><td>51.066</td><td>0.023</td><td>0.670</td><td>4.720</td><td>24.314</td></tr></table>

clock, are shown in Fig. 8, with a different color map from the dense LFs for distinction. It can be seen that our disparity maps have better visual qualities with smaller errors compared to the other methods. 

3) Evaluation on Real-world LF Images: We further evaluated different methods on the real-world LF images from the datasets [44]–[46]. All the methods were trained only on the synthetic LF images. Fig. 9 presents several predicted disparity maps of some methods. As the real-world scenes do not have ground truths, we can only compare their visual qualities. It can be seen that our method achieves more smooth predictions while preserving better object details, and therefore demonstrates better robustness and generalization compared to the other methods. 

4) Efficiency: We list the number of parameters and run time (the average time for outputting the disparity map for one LF image) of different methods, shown in Table IV 2. All the methods were implemented on a NVIDIA Tesla P100 GPU. It can be seen that our network is lightweight, and achieves a good balance between the accuracy and efficiency. 

# D. Ablation Studies

1) Design of DispNet: We first validated the coarse-to-fine structure of DispNet by training an additional model without the branch of residual estimation. We chose several scenes 

2UnPlug [28] employs the architecture of EPINet [21], so their number of parameters and run time are the same. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/de9652fe9470858f155f9fe4c41a0418bddbb6aaaaca0c3d29830ff625de7317.jpg)



Fig. 7. Visual results of different methods on the scenes from the DLF dataset [22], with the error maps relative to the ground truths.



TABLE II EVALUATION ON THE SCENES FROM DLF DATASET. BOLD: BEST AMONG THE UNSUPERVISED METHODS.


<table><tr><td rowspan="2">Methods</td><td colspan="4">Toys</td><td colspan="4">Antiques</td><td colspan="4">Pinenuts white</td><td colspan="4">Smiling crowd roses</td></tr><tr><td>MSE↓(×100)</td><td>BPR↓(0.07)</td><td>BPR↓(0.03)</td><td>BPR↓(0.01)</td><td>MSE↓(×100)</td><td>BPR↓(0.07)</td><td>BPR↓(0.03)</td><td>BPR↓(0.01)</td><td>MSE↓(×100)</td><td>BPR↓(0.07)</td><td>BPR↓(0.03)</td><td>BPR↓(0.01)</td><td>MSE↓(×100)</td><td>BPR↓(0.07))</td><td>BPR↓(0.03)</td><td>BPR↓(0.01)</td></tr><tr><td colspan="17">Supervised</td></tr><tr><td>EPINet [21]</td><td>0.431</td><td>15.540</td><td>42.438</td><td>75.800</td><td>1.265</td><td>6.992</td><td>32.292</td><td>72.562</td><td>0.509</td><td>15.232</td><td>35.826</td><td>69.557</td><td>3.148</td><td>14.997</td><td>41.297</td><td>77.024</td></tr><tr><td>LFAntNet [23]</td><td>0.405</td><td>10.231</td><td>35.711</td><td>74.166</td><td>0.827</td><td>4.206</td><td>21.862</td><td>67.076</td><td>0.406</td><td>10.698</td><td>27.042</td><td>65.995</td><td>2.025</td><td>12.454</td><td>33.356</td><td>66.702</td></tr><tr><td colspan="17">Unsupervised</td></tr><tr><td>UnCNN [25]</td><td>0.960</td><td>20.031</td><td>53.104</td><td>82.405</td><td>4.876</td><td>21.944</td><td>56.391</td><td>84.684</td><td>2.099</td><td>35.955</td><td>65.235</td><td>89.028</td><td>5.143</td><td>32.653</td><td>62.062</td><td>86.544</td></tr><tr><td>UnMonocular [26]</td><td>0.859</td><td>8.783</td><td>47.252</td><td>82.033</td><td>4.551</td><td>7.073</td><td>21.986</td><td>63.595</td><td>0.803</td><td>14.146</td><td>39.468</td><td>74.769</td><td>6.257</td><td>13.522</td><td>28.691</td><td>73.238</td></tr><tr><td>UnCon [29]</td><td>0.886</td><td>12.238</td><td>57.656</td><td>84.407</td><td>3.322</td><td>10.859</td><td>43.703</td><td>80.262</td><td>0.540</td><td>14.402</td><td>54.013</td><td>84.173</td><td>3.916</td><td>14.595</td><td>31.483</td><td>72.385</td></tr><tr><td>UnOcc [27]</td><td>1.021</td><td>18.497</td><td>44.714</td><td>76.093</td><td>4.034</td><td>7.018</td><td>22.748</td><td>61.508</td><td>0.706</td><td>21.065</td><td>49.413</td><td>79.825</td><td>3.678</td><td>15.403</td><td>32.678</td><td>71.964</td></tr><tr><td>Ours</td><td>0.643</td><td>6.745</td><td>24.210</td><td>60.513</td><td>2.336</td><td>5.514</td><td>12.937</td><td>42.088</td><td>0.434</td><td>8.106</td><td>25.164</td><td>61.817</td><td>3.243</td><td>11.325</td><td>23.254</td><td>50.507</td></tr></table>

from the HCI and DLF datasets for evaluation. Table V lists the number of parameters (during inference) and quantitative results of different configurations. It can be seen that the quantitative results have obvious decline if the coarse-to-fine structure is not employed, which suggests that the refinement branch with finer samplings helps to improve the disparity accuracy. Moreover, the refinement branch only introduces extra $\mathrm { 0 . 0 0 3 M }$ parameters since the feature extractor and the cost filters are all shared by the two branches. The visual comparison in Fig. 11 suggests that the DispNet with a coarse-to-fine structure achieves better disparity estimation with smaller errors. 

We also trained an additional model without shared weights for the cost filters. From Table V, we can see that separate cost filters for the two branches lead to more parameters but no improvement on the performance. Therefore, we chose to 

use shared cost filters in our DispNet. 

2) Occlusion Prediction: To verify the effectiveness of our OccNet for occlusion prediction, we trained an additional model by removing the OccNet and the loss terms $\ell _ { \mathrm { r e c } }$ and $\ell _ { \mathrm { s m o } }$ in Eq. 14. Thus, the pixel-wise weight of the photometric loss in Eq. 10 was reduced to 0.5. From Table V, we observe that the performance is degraded if the OccNet is not employed. The visual comparison in Fig. 11 reflects that the disparity map with occlusion prediction has smaller errors. 

The predicted occlusion maps for a synthetic dense LF, a real-world LF and a synthetic sparse LF are presented in Fig. 10. The occlusion regions are usually near the object boundaries [8], which leads to larger or smaller values within these regions in the occlusion maps. The red pixels with values approximate to 1 denote larger contributions for the reconstruction of the central view and also larger weights for 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/83106a9f205afe0f2e786d766c85d24ae0197101fb0188e0a08983a28808f0d2.jpg)



Fig. 8. Visual results of different methods on the sparse LF images from the SLF dataset [22], with the error maps relative to the ground truths.



TABLE III EVALUATION ON THE SCENES FROM SLF DATASET. BOLD: BEST AMONG THE UNSUPERVISED METHODS.


<table><tr><td rowspan="2">Methods</td><td colspan="4">Lion</td><td colspan="4">Rooster clock</td><td colspan="4">Toy bricks</td><td colspan="4">Electro devices</td></tr><tr><td>MSE↓</td><td>BPR↓(0.3)</td><td>BPR↓(0.1)</td><td>BPR↓(0.05)</td><td>MSE↓</td><td>BPR↓(0.3)</td><td>BPR↓(0.1)</td><td>BPR↓(0.05)</td><td>MSE↓</td><td>BPR↓(0.3)</td><td>BPR↓(0.1)</td><td>BPR↓(0.05)</td><td>MSE↓</td><td>BPR↓(0.3)</td><td>BPR↓(0.1)</td><td>BPR↓(0.05)</td></tr><tr><td colspan="17">Supervised</td></tr><tr><td>EPINet [21]</td><td>0.476</td><td>31.457</td><td>68.138</td><td>82.954</td><td>0.740</td><td>36.868</td><td>71.562</td><td>85.147</td><td>1.057</td><td>34.658</td><td>61.957</td><td>77.491</td><td>1.112</td><td>39.809</td><td>64.455</td><td>81.793</td></tr><tr><td>LFAttNet [23]</td><td>0.372</td><td>12.994</td><td>29.594</td><td>49.327</td><td>0.278</td><td>6.831</td><td>25.813</td><td>50.920</td><td>0.738</td><td>11.607</td><td>29.018</td><td>52.576</td><td>0.703</td><td>13.819</td><td>38.731</td><td>60.993</td></tr><tr><td colspan="17">Unsupervised</td></tr><tr><td>UnCNN [25]</td><td>1.442</td><td>58.221</td><td>84.867</td><td>92.373</td><td>9.213</td><td>34.586</td><td>75.021</td><td>87.383</td><td>2.254</td><td>53.922</td><td>82.172</td><td>90.950</td><td>2.181</td><td>41.163</td><td>72.970</td><td>85.866</td></tr><tr><td>UnMonocular [26]</td><td>0.760</td><td>48.305</td><td>87.371</td><td>94.673</td><td>0.664</td><td>39.931</td><td>73.601</td><td>86.370</td><td>1.226</td><td>47.594</td><td>71.098</td><td>84.982</td><td>2.736</td><td>45.612</td><td>75.238</td><td>87.332</td></tr><tr><td>UnCon [29]</td><td>1.192</td><td>35.493</td><td>75.392</td><td>87.891</td><td>0.491</td><td>16.928</td><td>53.889</td><td>75.597</td><td>1.741</td><td>50.471</td><td>79.470</td><td>89.474</td><td>3.604</td><td>42.443</td><td>75.654</td><td>87.271</td></tr><tr><td>UnOcc [27]</td><td>1.454</td><td>53.460</td><td>80.650</td><td>90.212</td><td>0.421</td><td>11.148</td><td>42.627</td><td>68.151</td><td>1.906</td><td>61.457</td><td>84.587</td><td>91.570</td><td>2.720</td><td>40.834</td><td>71.622</td><td>84.935</td></tr><tr><td>Ours</td><td>0.360</td><td>8.766</td><td>24.420</td><td>47.911</td><td>0.261</td><td>5.796</td><td>25.303</td><td>48.668</td><td>0.772</td><td>10.048</td><td>30.427</td><td>55.732</td><td>0.842</td><td>13.631</td><td>36.038</td><td>57.880</td></tr></table>


TABLE IV NUMBER OF PARAMETERS AND RUN TIME OF DIFFERENT METHODS


<table><tr><td>Method</td><td>Param. (M)</td><td>Run time (s)</td></tr><tr><td>EPINet [21]</td><td>2.466</td><td>2.765</td></tr><tr><td>LFAttNet [23]</td><td>5.540</td><td>6.479</td></tr><tr><td>UnCNN [25]</td><td>1.315</td><td>5.768</td></tr><tr><td>UnMonocular [26]</td><td>1.878</td><td>1.434</td></tr><tr><td>UnOcc [27]</td><td>1.891</td><td>1.840</td></tr><tr><td>UnCon [29]</td><td>2.185</td><td>2.102</td></tr><tr><td>UnPlug [28]</td><td>2.466</td><td>2.765</td></tr><tr><td>Ours</td><td>1.802</td><td>2.688</td></tr></table>

the photometric loss. The case is opposite for the blue pixels with values approximate to 0. Moreover, due to the larger disparity range, the sparse LFs usually have larger occlusion 


TABLE V ABLATION STUDY ON THE FRAMEWORK. BOLD: BEST.


<table><tr><td>Configuration</td><td>Param. (M)</td><td>MSE↓ (×100)</td><td>BPR↓ (0.07)</td><td>BPR↓ (0.03)</td><td>BPR↓ (0.01)</td></tr><tr><td>w/o coarse-to-fine</td><td>1.799</td><td>3.057</td><td>11.487</td><td>27.302</td><td>62.725</td></tr><tr><td>w/o shared weights</td><td>2.230</td><td>2.282</td><td>9.365</td><td>21.882</td><td>54.137</td></tr><tr><td>w/o OccNet</td><td>1.802</td><td>2.701</td><td>10.806</td><td>24.466</td><td>58.244</td></tr><tr><td>Default</td><td>1.802</td><td>2.266</td><td>9.238</td><td>21.683</td><td>54.042</td></tr></table>

regions, resulting in much thicker red and blue regions near the object boundaries compared to the dense LFs. 

3) Loss Terms: We trained additional models by excluding the SSIM loss $\ell _ { \mathrm { S S I M } }$ , the smoothness loss for disparity map $\ell _ { \mathrm { s m d } }$ and the smoothness loss for occlusion map $\ell _ { \mathrm { s m o } }$ , since $\ell _ { \mathrm { w p m } }$ and $\ell _ { \mathrm { r e c } }$ are indispensable to the training of DispNet 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/1eedc1004613b3943c89ea3f5560c49d35776eb9e02955168388b7dfbe880d9e.jpg)



Fig. 9. Visual results of different methods on the real-world LF images from the datasets [44]–[46]. Zoom in for best view.



TABLE VI ABLATION STUDY ON THE LOSS TERMS. BOLD: BEST.


<table><tr><td>Loss terms</td><td>MSE↓ (×100)</td><td>BPR↓ (0.07)</td><td>BPR↓ (0.03)</td><td>BPR↓ (0.01)</td></tr><tr><td>w/o ℓSSIM</td><td>2.393</td><td>9.535</td><td>22.027</td><td>55.250</td></tr><tr><td>w/o ℓsmd</td><td>2.457</td><td>9.749</td><td>22.136</td><td>55.856</td></tr><tr><td>w/o ℓsmo</td><td>2.276</td><td>9.884</td><td>22.304</td><td>54.275</td></tr><tr><td>Full loss</td><td>2.266</td><td>9.238</td><td>21.683</td><td>54.042</td></tr></table>


TABLE VII ABLATION STUDY ON DISPARITY FUSION APPROACHES. BOLD: BEST.


<table><tr><td>Fusion strategy</td><td>MSE↓(×100)</td><td>BPR↓(0.07)</td><td>BPR↓(0.03)</td><td>BPR↓(0.01)</td></tr><tr><td>w/o occlusion handling</td><td>2.837</td><td>10.371</td><td>23.441</td><td>56.778</td></tr><tr><td>Minimum error fusion</td><td>2.411</td><td>10.210</td><td>23.743</td><td>57.842</td></tr><tr><td>Weighted fusion (n′ = 4)</td><td>2.564</td><td>10.022</td><td>22.513</td><td>54.462</td></tr><tr><td>Weighted fusion (n′ = 2)</td><td>2.266</td><td>9.238</td><td>21.683</td><td>54.042</td></tr></table>

and OccNet. Table VI lists the corresponding results, which suggests that the MSE and BPRs slightly increase after removing each of them. Therefore, these loss terms contributes to a better performance. 

4) Disparity Fusion Strategy: We first evaluated different disparity fusion strategies, and our default strategy is the weighted fusion by using two disparities $\mathit { \Delta } n ^ { \prime } = 2 \mathit { \Delta }$ with the smallest errors at each position. The other fusion approaches include the weighted fusion by using all the disparities $n ^ { \prime } =$ 4), the minimum error fusion that selects the disparity with the minimum error at each position. We also list the results of no occlusion handling when deriving the error maps (the estimated error of each position is represented by the mean of all the warping errors) with weighted fusion $\left( n ^ { \prime } \ = \ 2 \right)$ . Their quantitative results are given in Table VII and the visual comparison is shown in Fig. 12, where we observe that the weighted fusion with $n ^ { \prime } = 2$ obtains better results than the other strategies, and the performance has obvious decline if occlusion is not handled during fusion. 

Another approach to address occlusion is to use the average value of the $k$ smallest warping errors. We experimented with $k = 1 , 4 , 8 , 1 0 , 1 2 , 1 6$ (totally 16 auxiliary views) using weighted fusion with $n ^ { \prime } = 2$ . Fig. 13 shows the quantitative results at different $k$ values. It can be seen that the results are best at $k \ = \ 8$ , but they are still worse than those of using median value as listed in Table VII. As large warping errors are mainly caused by the occlusion and inaccurate disparity estimation, large $k$ value may lead to incomplete 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/eed0faacd25d72dc75f4512224171b279a952b3f603484eda597a05e0aa20a0f.jpg)



Fig. 10. The predicted occlusion maps of a synthetic dense LF, a real-world LF and a synthetic sparse LF.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/7ed297a31d83f2b9186753a3ae634cab3ad1d7c30434454d90ec46b7b1c74bc4.jpg)



Fig. 11. Visual comparison of different network configurations. Zoom in for best view.


occlusion handling while small $k$ value may lead to inaccurate error estimation at the pixels with inaccurate disparity values. Therefore, using median value is more effective to address these issues. 

We then investigated the effect of quantile $q$ for determining the threshold of occlusion. We experimented with $q = 0 . 8 , 0 . 8 5 , 0 . 9 , 0 . 9 5 , 0 . 9 8 , 1 . q = 0 . 9 5$ $q = 0 . 9 5$ means that $5 \%$ pixels with the largest standard deviations use the median value of the warping errors to address the occlusion issue, and $q = 1$ means that all the pixels use the mean value of the warping errors, which is equivalent to the strategy without occlusion handling in Table VII. Fig. 14 shows the effect of $q$ on the quantitative results, where we can see that $q = 0 . 9 5$ obtains better results than the other values, and the MSE increases significantly at $q = 0 . 9 8$ and $q = 1$ , verifying the effectiveness of occlusion handling. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/5dae2ab2171588bf853241b100aeafe0cff097a3c34618aa37cfa672f6949715.jpg)



Fig. 12. Visual comparison of different disparity fusion strategies. Zoom in for best view.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/4059d23b781e27998a1a253267ae5b6e541593f1d630714497ee08b4a3f2599d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/6a14377ab6b4a5d1f2f00115a84b5fd768678da8755f929f0dda2b1b04994d55.jpg)



Fig. 13. The effect of $k$ on the quantitative results.


# V. CONCLUSION

This paper presents an unsupervised framework for LF depth estimation. We first develop a DispNet that takes as inputs different view combinations to predict multiple disparity maps. It adopts a coarse-to-fine structure with two branches to estimate the coarse and residual disparity maps, respectively, through multi-view feature matching. Photo-consistency is the main supervisory signal for the unsupervised training. However, it does not hold in the occlusion regions. To tackle this issue, we introduce an OccNet to predict the occlusion maps, which are used as the element-wise weights of the photometric loss to alleviate the impact of occlusions on the disparity learning. The OccNet is trained by the reconstruction loss with no ground truth required. The multiple estimated disparity maps after rotating and scaling are fused based on their estimated errors using the auxiliary views with effective occlusion handling to obtain the final disparity map. Our framework achieves superior performance on both the dense and sparse LF images, and also demonstrates better generalization ability on the real-world LF images. 

# REFERENCES



[1] R. Ng, M. Levoy, M. Bredif, G. Duval, M. Horowitz, and P. Hanrahan, ´ “Light field photography with a hand-held plenoptic camera,” Computer Science Technical Report, vol. 2, no. 11, pp. 1–11, 2005. 





[2] E. Y. Lam, “Computational photography with plenoptic camera and light field capture: tutorial,” Journal of the Optical Society of America A, vol. 32, no. 11, pp. 2021–2032, 2015. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/2e09cdc2fc58a607f3ea19220ffe2fe8632e53833c9ac0ed4510c5d732b282ba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/f2d0fc3409291d8d23bd83be7d38c26038b3631683fa88bbf27617a6f3dd44f6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/6f256ec791af16b6799bfa5d871a759969fbc701baec1ed498c6735195aed4ed.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-23/140b46e8-4ae2-4fcd-9e43-2a9d4d6c77e5/7c39cc023351eb645754ab3e7fc6b9536e7c013b524c63e309f4917d1b674889.jpg)



Fig. 14. The effect of $q$ on the quantitative results.




[3] S. Zhang and E. Y. Lam, “A deep retinex framework for light field restoration under low-light conditions,” in Proceedings of the International Conference on Pattern Recognition, 2022, pp. 2042–2048. 





[4] S. Zhang, N. Meng, and E. Y. Lam, “LRT: An efficient low-light restoration transformer for dark light field images,” IEEE Transactions on Image Processing, vol. 32, pp. 4314-4326, 2023. 





[5] J. Fiss, B. Curless, and R. Szeliski, “Refocusing plenoptic images using depth-adaptive splatting,” in Proceedings of the IEEE International Conference on Computational Photography, 2014, pp. 1–9. 





[6] C. Kim, H. Zimmer, Y. Pritch, A. Sorkine-Hornung, and M. H. Gross, “Scene reconstruction from high spatio-angular resolution light field,” ACM Transactions on Graphics, vol. 32, no. 4, 2013. 





[7] J. Jin, J. Hou, H. Yuan, and S. Kwong, “Learning light field angular super-resolution via a geometry-aware network,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2020, pp. 11141–11148. 





[8] N. Meng, K. Li, J. Liu, and E. Y. Lam, “Light field view synthesis via aperture disparity and warping confidence map,” IEEE Transactions on Image Processing, vol. 30, pp. 3908–3921, 2021. 





[9] J. Chen and L.-P. Chau, “Light field compressed sensing over a disparityaware dictionary,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 27, no. 4, pp. 855–865, 2017. 





[10] H. Sheng, R. Cong, D. Yang, R. Chen, S. Wang, and Z. Cui, “UrbanLF: A comprehensive light field dataset for semantic segmentation of urban scenes,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 11, pp. 7880–7893, 2022. 





[11] S. Wanner and B. Goldluecke, “Variational light field analysis for disparity estimation and super-resolution,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 3, pp. 606–619, 2014. 





[12] S. Zhang, H. Sheng, C. Li, J. Zhang, and Z. Xiong, “Robust depth estimation for light field via spinning parallelogram operator,” Computer Vision and Image Understanding, vol. 145, pp. 148–159, 2016. 





[13] Y. Zhang, H. Lv, Y. Liu, H. Wang, X. Wang, Q. Huang, X. Xiang, and Q. Dai, “Light-field depth estimation via epipolar plane image analysis and locally linear embedding,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 27, no. 4, pp. 739–747, 2017. 





[14] H. Sheng, P. Zhao, S. Zhang, J. Zhang, and D. Yang, “Occlusion-aware depth estimation for light field using multi-orientation EPIs,” Pattern Recognition, vol. 74, pp. 587–599, 2018. 





[15] H.-G. Jeon, J. Park, G. Choe, J. Park, Y. Bok, Y.-W. Tai, and I. So Kweon, “Accurate depth map estimation from a lenslet light field camera,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 





[16] M. W. Tao, P. P. Srinivasan, J. Malik, S. Rusinkiewicz, and R. Ramamoorthi, “Depth from shading, defocus, and correspondence using light-field angular coherence,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 





[17] J. Y. Lee and R.-H. Park, “Depth estimation from light field by accumulating binary maps based on foreground–background separation,” IEEE Journal of Selected Topics in Signal Processing, vol. 11, no. 7, pp. 955–964, 2017. 





[18] C.-T. Huang, “Empirical bayesian light-field stereo matching by robust pseudo random field modeling,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 3, pp. 552–565, 2019. 





[19] X. Sun, Z. Xu, N. Meng, E. Y. Lam, and H. K.-H. So, “Data-driven light field depth estimation using deep convolutional neural networks,” in Proceedings of the International Joint Conference on Neural Networks, 2016, pp. 367–374. 





[20] S. Heber, W. Yu, and T. Pock, “Neural EPI-volume networks for shape from light field,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017. 





[21] C. Shin, H.-G. Jeon, Y. Yoon, I. S. Kweon, and S. J. Kim, “EPINET: A fully-convolutional neural network using epipolar geometry for depth from light field images,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 





[22] J. Shi, X. Jiang, and C. Guillemot, “A framework for learning depth from a flexible subset of dense and sparse light field views,” IEEE Transactions on Image Processing, vol. 28, no. 12, pp. 5867–5880, 2019. 





[23] Y.-J. Tsai, Y.-L. Liu, M. Ouhyoung, and Y.-Y. Chuang, “Attentionbased view selection networks for light-field disparity estimation,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2020. 





[24] J. Chen, S. Zhang, and Y. Lin, “Attention-based multi-level fusion network for light field depth estimation,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2021. 





[25] J. Peng, Z. Xiong, D. Liu, and X. Chen, “Unsupervised depth estimation from light field using a convolutional neural network,” in Proceedings of the International Conference on 3D Vision, 2018, pp. 295–303. 





[26] W. Zhou, E. Zhou, G. Liu, L. Lin, and A. Lumsdaine, “Unsupervised monocular depth estimation from light field image,” IEEE Transactions on Image Processing, vol. 29, pp. 1606–1617, 2020. 





[27] J. Jin and J. Hou, “Occlusion-aware unsupervised learning of depth from 4-d light fields,” IEEE Transactions on Image Processing, vol. 31, pp. 2216–2228, 2022. 





[28] T. Iwatsuki, K. Takahashi, and T. Fujii, “Unsupervised disparity estimation from light field using plug-and-play weighted warping loss,” Signal Processing: Image Communication, vol. 107, p. 116764, 2022. 





[29] L. Lin, Q. Li, B. Gao, Y. Yan, W. Zhou, and E. E. Kuruoglu, “Unsupervised learning of light field depth estimation with spatial and angular consistencies,” Neurocomputing, vol. 501, pp. 113–122, 2022. 





[30] T.-H. Tran, G. Mammadov, and S. Simon, “GVLD: A fast and accurate GPU-based variational light-field disparity estimation approach,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 31, no. 7, pp. 2562–2574, 2021. 





[31] N. Meng, H. K.-H. So, X. Sun, and E. Y. Lam, “High-dimensional dense residual convolutional neural network for light field reconstruction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 3, pp. 873–886, 2021. 





[32] X. Wang, J. Liu, S. Chen, and G. Wei, “Effective light field de-occlusion network based on swin transformer,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 6, pp. 2590–2599, 2023. 





[33] Y. Zhang, W. Dai, M. Xu, J. Zou, X. Zhang, and H. Xiong, “Depth estimation from light field using graph-based structure-aware analysis,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 30, no. 11, pp. 4269–4283, 2020. 





[34] T.-C. Wang, A. A. Efros, and R. Ramamoorthi, “Depth estimation with occlusion modeling using light-field cameras,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 11, pp. 2170– 2181, 2016. 





[35] W. Williem and I. K. Park, “Robust light field depth estimation for noisy scene with occlusion,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 





[36] J. Chen, J. Hou, Y. Ni, and L.-P. Chau, “Accurate light field depth estimation with superpixel regularization over partially occluded regions,” IEEE Transactions on Image Processing, vol. 27, no. 10, pp. 4889–4900, 2018. 





[37] Y. Wang, L. Wang, Z. Liang, J. Yang, W. An, and Y. Guo, Occlusionaware cost constructor for light field depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 19809–19818. 





[38] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam, “Rethinking atrous convolution for semantic image segmentation,” arXiv preprint arXiv: 1706.05587, 2017. 





[39] J.-R. Chang and Y.-S. Chen, “Pyramid stereo matching network,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 





[40] S. Zhang and E. Y. Lam, “Learning to restore light fields under low-light imaging”, Neurocomputing, vol. 456, pp. 76–87, 2021. 





[41] C. Godard, O. MacAodha, and G. J. Brostow, “Unsupervised monocular depth estimation with left-right consistency,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 





[42] S. Zhang and E. Y. Lam, “An effective decomposition-enhancement method to restore light field images captured in the dark,” Signal Processing, vol. 189, p. 108279, 2021. 





[43] K. Honauer, O. Johannsen, D. Kondermann, and B. Goldluecke, “A Dataset and Evaluation Methodology for Depth Estimation on 4D Light Fields,” in Proceedings of the Asian Conference on Computer Vision, 2016, pp. 19–34. 





[44] R. Shah, G. Wetzstein, A. S. Raj, and M. Lowney, “Stanford Lytro light field archive,” 2016. [Online]. Available: http://lightfields.stanford.edu/ LF2016.html 





[45] N. K. Kalantari, T. C. Wang, and R. Ramamoorthi, “Learning-based view synthesis for light field cameras,” ACM Transactions on Graphics, vol. 35, no. 6, 2016. 





[46] M. Rerabek and T. Ebrahimi, “New light field image dataset,” in Proceedings of the International Conference on Quality of Multimedia Experience, 2016. 

