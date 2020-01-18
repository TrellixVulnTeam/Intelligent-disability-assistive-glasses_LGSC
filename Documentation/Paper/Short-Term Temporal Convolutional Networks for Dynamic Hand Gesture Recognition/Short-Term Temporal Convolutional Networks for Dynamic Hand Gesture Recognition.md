# Short-Term Temporal Convolutional Networks forDynamic Hand Gesture Recognition
# 短期时间卷积网络的动态手势识别

Yi Zhang∗, Chong Wang†, Ye Zheng, Jieyu Zhao, Yuqi Li and Xijiong Xie‡
January 17, 20

## Abstract
## 摘要

The purpose of gesture recognition is to recognize meaningful movements of human bodies, and gesture recognition is an important issue in computer vision. 

手势识别的目的是识别人体有意义的运动，而手势识别是计算机视觉中的重要问题。

In this paper, we present a multimodal gesture recognition method based on 3D densely convolutional networks (3D-DenseNets) and improved temporal convolutional networks (TCNs). 

在本文中，我们提出了一种基于3D密集卷积网络（3D-DenseNets）和改进的时间卷积网络（TCNs）的多模式手势识别方法。

The key idea of our approach is to find a compact and effective representation of spatial and temporal features, which orderly and separately divide task of gesture video analysis into two parts: spatial analysis and temporal analysis.

我们方法的关键思想是找到一种紧凑有效的空间和时间特征表示，将手势视频分析的任务有序且分别地分为空间分析和时间分析两部分。

In spatial analysis, we adopt 3D-DenseNets to learn short-term spatiotemporal features effectively. 

在空间分析中，我们采用3D-DenseNets有效地学习短期时空特征。

Subsequently, in temporal analysis, we use TCNs to extract temporal features and employ improved Squeeze-andExcitation Networks (SENets) to strengthen the epresentational power of temporal features from each TCNs’ layers. 

随后，在时间分析中，我们使用TCN提取时间特征，并使用改进的挤压和激发网络（SENets）来增强每个TCN层的时间特征的表示能力。

The method has been evaluated on the VIVA and the NVIDIA Gesture Dynamic Hand Gesture Datasets.

该方法已在VIVA和NVIDIA Gesture动态手势数据集上进行了评估。

Our approach obtains very competitive performance on VIVA benchmarks with the classification accuracies of 91.54%, and achieve stateof-the art performance with 6.37% accuracy on NVIDIA benchmark.

我们的方法在VIVA基准上获得了非常有竞争力的性能，分类精度为91.54％，在NVIDIA基准上达到了6.37％的准确性。

**Index terms— Gesture Recognition, 3D-DenseNets, TCNs, multimodal.**
**索引词-手势识别，3D-DenseNet，TCN，多模式。**

## 1 Introduction
## 1 引言

Gesture recognition is a fast expanding field with applications in human-computer interaction[1], sign language recognition[2] and etc.. 

手势识别是一个快速发展的领域，已在人机交互[1]，手语识别[2]等领域得到应用。

Due to subtle differences among similar gestures, complex scene background, different observation conditions, and noises in acquisition, robust gesture recognition is very challenging.

由于相似手势之间的细微差异，复杂的场景背景，不同的观察条件以及采集中的噪音，鲁棒的手势识别非常具有挑战性。

The main task of gesture recognition is to extract features from an image or a video and then classify or determine each sample to a certain label. 

手势识别的主要任务是从图像或视频中提取特征，然后将每个样本分类或确定为某个标签。

Gesture recognition aims to recognize and understand meaningful movement of human bodies in which arms and hands play crucial roles. 

手势识别旨在识别和理解有意义的人体运动，其中手臂和手起着至关重要的作用。

Only few gestures can be identified from their spatial or structure information in an image or a single frame. 

从图像或单个帧中的空间或结构信息中只能识别出很少的手势。

In fact, motion cues and structure information simultaneously characterize a unique gesture. 

实际上，运动提示和结构信息同时表征了唯一手势。

How to learn spatiotemporal features effectively is always the key in gesture recognition.

如何有效地学习时空特征一直是手势识别的关键。

Although in the past decades, many methods have been proposed for this issue, ranging from static to dynamic gestures, and from motion silhouettes-based to the convolutional neural network-based, there are still many challenges associated with the recognition accuracy.

尽管在过去的几十年中，针对该问题已提出了许多方法，从静态手势到动态手势，从基于运动轮廓的轮廓到基于卷积神经网络的轮廓，但仍存在许多与识别准确性相关的挑战。

At present, although most existing models have reached a high performance for isolated gesture recognition, most methods have been developed based on Convolutional Neural Networks (CNNs)[3][4] or Recurrent Neural Networks (RNNs)[5]. 

目前，尽管大多数现有模型已经实现了隔离手势识别的高性能，但是大多数方法都是基于卷积神经网络（CNN）[3] [4]或递归神经网络（RNN）[5]开发的。

With the development of deep learning, more and more new architectures of CNNs have been proposed, especially DenseNets [6] what have powerful feature extraction ability. 

随着深度学习的发展，提出了越来越多的CNN新架构，尤其是具有强大特征提取能力的DenseNets [6]。

Meanwhile, a new architecture to solve sequence problem named TCNs [7] have been proposed. 

同时，提出了一种解决序列问题的新架构，称为TCN [7]。

Compared to RNNs and their canonical recurrent architectures such as LSTMs and GRUs, TCNs have comparable clarity and simplicty. 

与RNN及其规范的递归体系结构（如LSTM和GRU）相比，TCN具有相当的清晰度和简单性。

In our approach, we adopt 3D-DenseNet to extract short-term stapio-temporal features, then these features are input into the TCNs to finish the task of Classification.

在我们的方法中，我们采用3D-DenseNet提取短期时针时态特征，然后将这些特征输入到TCN中以完成分类任务。

However, recently, for extracting more complete temporal features, a few methods have been proposed based on attention mechanism. 

但是，最近，为了提取更完整的时间特征，基于注意力机制提出了一些方法。

The research prove that there are various relationships between features’ interior in neural networks.

研究证明，神经网络中特征内部之间存在各种关系。

SENets [8] are new architectural unit with the goal of improving the quality of representations produced by a network by explicitly modelling the interdependencies between the channels of its convolutional features. 

SENets [8]是一个新的体系结构单元，其目标是通过显式地建模其卷积特征的通道之间的相互依赖性来提高网络产生的表示的质量。

And in our approach, we reform SENets and combine them into TCNs to strengthen capacity of TCNs in temporal features extracting.

在我们的方法中，我们对SENet进行了改革，并将其组合为TCN，以增强TCN在时域特征提取中的能力。

The pipline of our method is depicted in Figure 1, and the main contribution can be summarized as following:

我们的方法的流程如图1所示，其主要贡献可总结如下：

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/figure1.png)

**Figure 1: An overview of the proposed method.**
**图1：建议方法的概述。**

**The proposed deep architecture is composed of two main steps:** 
**拟议的深度架构包括两个主要步骤：**

**(a) Multimodal short-term spatio-temporal feature sequence extracting by truncated 3D-Densenet (T3D-Dense), local temporal average pooling (LTAP) and multimodal features concatenation.**
**（a）通过截断的3D-Densenet（T3D-Dense），局部时间平均池（LTAP）和多峰特征级联来提取多峰短期时空特征序列。**

**(b)Longterm feature sequence recognizing via TCN and TSE.**
**（b）通过TCN和TSE识别长期特征序列。**

<p>• Spatial analysis.</p>
<p>•空间分析。</p>

We design a multi-stream truncated 3D-DenseNet, which extracts spatio-temporal features from a video, and through local temporal pooling, obtain the decomposed short-term spatio-temporal features, to solve problem that single frame image can not carry enough spatial or structure information of gesture and reduce repetitive training for video clips.

我们设计了一种多流截断的3D-DenseNet，该方法从视频中提取时空特征，并通过局部时域池化，获得分解后的短期时空特征，以解决单帧图像不能承载足够空间空间的问题。 或构造手势信息，并减少对视频剪辑的重复训练。

<p>• Temporal analysis. </p>
<p>•时间分析。</p>

We employ TCN to replace RNN as the main model of sequence information feature analysis. 
我们采用TCN代替RNN作为序列信息特征分析的主要模型。

In addition, we improve SENets and apply them in temporal domain to rescale the weights between temporal features and extract more effective temporal features to achieve higher classification accuracy.
此外，我们改进了SENet，并将其应用于时域以重新调整时态特征之间的权重，并提取更有效的时态特征以实现更高的分类精度。

## 2 Related Work
## 2 相关工作

Gesture taxonomies and representations have been studied for decades.

手势分类法和表示法已经研究了数十年。

The vision based gesture recognition techniques include the static gesture oriented and the dynamic gesture oriented methods [1].

基于视觉的手势识别技术包括面向静态手势和面向动态手势的方法[1]。

Recently, convolution neural networks (CNNs) [9] have made a great breakthrough on computer vision related tasks by their powerful feature extraction ability, thus the features extracted by CNNs are widely used in many action classification tasks instead of hand-crafted features for better performance. 

近年来，卷积神经网络（CNN）[9]以其强大的特征提取能力在计算机视觉相关任务上取得了重大突破，因此，CNN提取的特征被广泛用于许多动作分类任务中，而不是手工制作的特征，从而获得更好的效果和性能。

Features are extracted by 2D-CNN from the starts. 

从一开始就由2D-CNN提取特征。

bi-directional rank pooling [10][11] was used to encode the spatial and temporal information of videos. 

双向秩合并[10] [11]用于编码视频的时空信息。

Temporal convolutions for gesture recognition in videos.

视频中的时间卷积用于手势识别。

Beyond temporal pooling [12] was proposed to solve gesture recognition problem in videos by a new temporal pooling method. 
提出了一种超越时间池的方法[12]，它通过一种新的时间池方法来解决视频中的手势识别问题。

On the other hand, C3D[13] model is developed and provides a better performance and main contribution in this research is proposed an architecture to extract spatio-temporal features from a video clip. 

另一方面，C3D [13]模型得到了发展，并提供了更好的性能，并且在这项研究中的主要贡献是提出了一种从视频剪辑中提取时空特征的体系结构。

Concurrently, a multi-stream 3D-CNN[14] was designed for hand gesture recognition and the classifier consisted of two subnetworks: a high-resolution network (HRN) and a low-resolution network (LRN) in this model.

同时，为手势识别设计了一个多流3D-CNN [14]，分类器由两个子网组成：该模型中的一个高分辨率网络（HRN）和一个低分辨率网络（LRN）。

Meanwhile, with the development of convolutional neural networks, more and more architectures of CNNs were proposed, like AlexNet [9], VGGNet [15], GoogleNet [16] [17] [18] [19], ResNet [20] and DenseNet [6]. 

同时，随着卷积神经网络的发展，提出了越来越多的CNN架构，例如AlexNet [9]，VGGNet [15]，GoogleNet [16] [17] [18] [19]，ResNet [20]和DenseNet [6]。

All of these models have one target that is building a higher architectures of CNNs to dig deeper and more complete statial features from low-level image frames, and then classify. 

所有这些模型都有一个目标，那就是构建更高的CNN架构，以从低级图像帧中挖掘更深，更完整的统计特征，然后进行分类。

In the area of isolated gesture recognition, Res-C3D model[21] was used and won the first place twice in ChaLearn LAP Multi-modal Isolated Gesture Recognition Challenges 2016 [22] and 2017 [23]. 

在孤立手势识别领域，使用了Res-C3D模型[21]，并在ChaLearn LAP多模式孤立手势识别挑战赛2016 [22]和2017 [23]中两次获得第一名。

Whatmore, DenseNets as one of the latest convolutional architectures, was adopted in action recognitions especially face recognitions and gesture recognitions gradually. 

此外，DenseNets作为最新的卷积架构之一，逐渐被用于动作识别，尤其是面部识别和手势识别。

A face recognition model named Dense Face[24] was proposed to explore the performance of densely connected network in face recognition. 

为了探究密集连接网络在人脸识别中的性能，提出了一种名为“ Dense Face [24]”的人脸识别模型。

DenseNets[25] also was used to classifier the different actions in recent researches.

在最近的研究中，DenseNets [25]也被用来分类不同的动作。

Regarding the temporal information of the video sequences, Long Short Term Memory(LSTM) networks is a common choice to gesture recognition. 

关于视频序列的时间信息，长短期记忆（LSTM）网络是手势识别的常见选择。

For instance, convolutional LSTM[26] was introduced for spatio-temporal feature maps. 

例如，针对时空特征图引入了卷积LSTM [26]。

2S-RNN(RGB and Depth)[27] was used for continuous gesture recognition. 

2S-RNN（RGB和深度）[27]用于连续手势识别。

However, RNNs including LSTMs and GRUs have some weaknesses on temporal domain like short-range information learning, oversized memory capacity. 

但是，包括LSTM和GRU在内的RNN在时域上有一些弱点，例如短距离信息学习，超大存储容量。

To make these weaknesses up, TCNs is proposed and applied in the gesture reconition. 

为了弥补这些弱点，提出了TCN，并将其应用于手势重构中。

Res-TCN[28] was proposed for skeleton-based dynamic hand gesture recognition. 

Res-TCN [28]被提出用于基于骨骼的动态手势识别。

Whatmore, a model based on TCN[29] was proposed for gesture recognition.

此外，提出了基于TCN [29]的手势识别模型。

Other important works based on attention mechanism. 

基于注意机制的其他重要著作。

Attention mechanism or attention model firstly was applied to neural networks by Vaswani et al[30]. 

Vaswani等[30]首先将注意力机制或注意力模型应用于神经网络。

After that, more and more researches are proposed based on attention mechanism, so as SENets [8] that improve ResNets to win first place of ILSVRC 2017 classification.

之后，基于注意力机制的研究越来越多，例如SENets [8]改进了ResNets以赢得ILSVRC 2017分类的第一名。

## 3 Our Approach
## 3 我们的方法

In the video recognition, both of the spatial and temporal information are important. 

在视频识别中，空间和时间信息都很重要。

Although there have been impressive progress in spatial feature extraction using 2D-CNNs based networks[14][3], how to effectively learn the temporal features is still a very challenging problem. 

尽管使用基于2D-CNN的网络在空间特征提取方面取得了令人瞩目的进展[14] [3]，但是如何有效地学习时间特征仍然是一个非常具有挑战性的问题。

Unlike the 2D-CNNs focusing on the single image, various 3D-CNN based networks[5][31][32][33] have been proposed to process the successive frames simultaneously. 

与专注于单个图像的2D-CNN不同，已经提出了各种基于3D-CNN的网络[5] [31] [32] [33]来同时处理连续帧。

For the video of dynamic hand gestures, adjacent frames are usually similar and containing the same static gesture, while the static gestures change several times during the whole video.

对于动态手势的视频，相邻帧通常相似并且包含相同的静态手势，而静态手势在整个视频中会多次更改。

Thus, in this paper we decompose the video to two different parts. 

因此，在本文中，我们将视频分解为两个不同的部分。

One is the short-term spatio-temporal information in the adjacent frames, and the other is the long-term temporal information analysed by a sequential model. 

一种是相邻帧中的短期时空信息，另一种是通过顺序模型分析的长期时空信息。

Based on this consideration, we raised two major questions, 

基于这种考虑，我们提出了两个主要问题，

• how to learn short-term spatio-temporal features effectively from video clips in the same video.

• 如何从同一视频中的视频片段中有效学习短期时空特征。

• how to reasonably classify a sequence which is combined from these consecutive features.

•如何合理地对由这些连续特征组合而成的序列进行分类。

In order to address these issues, we designed a novel architecture to extract a sequence of short spatiotemporal features in order to recognize dynamic gestures.

为了解决这些问题，我们设计了一种新颖的架构来提取一系列短时空特征以识别动态手势。

As depicted in Figure 1, the overall process can be divided into two parts:

如图1所示，整个过程可以分为两部分：

1) multi-modal short-term spatio-temporal feature extraction based on 3DDenseNets and 2) spatio-temporal sequence classify with and temporal SENets embedded TCNs. 

1）基于3DDenseNets的多模式短期时空特征提取，以及2）带有时态SENets嵌入式TCN的时空序列分类。

To be specific, the details of the proposed network structure is presented in Figure ?? and Figure ??.

具体来说，建议的网络结构的详细信息如图所示。

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/figure2.png)

**Figure 2: The architecture of 3D-DenseNet.**

**图2：3D-DenseNet的体系结构。**


### 3.1 temporal local pooling to extract short-term features
### 3.1 时间局部池提取短期特征

Due to the availability of various data types and the nature of signing videos, a more robust feature representation can acquired from the incorporation of multimodal hand gesture information. 

由于各种数据类型的可用性和签名视频的性质，可以通过合并多模式手势信息来获得更强大的功能表示。

To effectively present the the location, shape and sequential information in the adjacent gesture frames, we design a multistream DenseNet based on the C3D[13] to extracts short-term spatio-temporal features. 

为了有效地显示相邻手势帧中的位置，形状和顺序信息，我们基于C3D [13]设计了一种多流DenseNet，以提取短期时空特征。

Assume a given video V with n frames, it is firstly resampled to k frames. 

假设给定的视频V具有n帧，则首先将其重新采样为k帧。

Thus, the input video VS is denoted as,

因此，输入视频VS表示为

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/VS.png)

where v<sub>k</sub> is the k-th frame image of video sequence in the input.

其中v<sub>k</sub>是输入中视频序列的第k个帧图像。

As aforementioned, we consider multiple modalities of gesture video data as the input. 

如前所述，我们将手势视频数据的多种形式视为输入。

Each type of the data is set as one data stream and fed to the same network structure. 

每种类型的数据都设置为一个数据流，并馈送到相同的网络结构。

The outputs of them will be fused together later as shown in Figure 1. 

它们的输出稍后将融合在一起，如图1所示。

The proposed model contains 4 dense blocks, containing 6, 12, 24, 16 layers respectively. 

所提出的模型包含4个密集块，分别包含6、12、24、16层。

Following the basic design in DenseNet[6] and C3D[13], the detailed network configurations are shown in Table 1. 

遵循DenseNet [6]和C3D [13]的基本设计，详细的网络配置如表1所示。

It is worth noting that most of the convolution layers are with 3 ×3×3 filters, which limits the process only on the local spatial and temporal domain. 

值得注意的是，大多数卷积层都带有3×3×3滤波器，这仅在局部时空域上限制了该过程。

Moreover, the temporal pooling size and stride in all the transition layers are set as 1 to avoid the fusion of the short-term temporal information, which is one major difference from the other conventional 3D-CNNs[13].

此外，将所有过渡层中的时间池大小和跨度设置为1，以避免短期时间信息的融合，这是与其他传统3D-CNN的主要区别[13]。

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/table1.png)

**Table 1: 3D-DenseNet architectures. **

**表1：3D-DenseNet架构。**

**The growth rate of network is k = 12.** 

**网络的增长率为k = 12。**

**Note that each “conv” layer shown in the corresponds the sequence BN-ReLU-Conv.**

**注意，所示的每个“ conv”层对应于序列BN-ReLU-Conv。**

Since the 3D-Densenet is served as a short-term spatio-temporal features extractor, we truncate it to obtain the features only. 

由于3D-Densenet用作短期时空特征提取器，因此我们将其截短以仅获取特征。

To be specific, the global temporal average pooling layer, last softmax and fully-connected layers are discarded, after the model is first pre-trained with isolated gesture data.

具体而言，在首先使用隔离的手势数据对模型进行预训练之后，将丢弃全局时间平均池化层，最后的softmax和完全连接的层。

Therefore, we can get the global spatio-temporal feature F<sub>k</sub> after the global spatial average pool layer,

因此，我们可以在全局空间平均池层之后获得全局时空特征F<sub>k</sub>，

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/FK.png)

Then T short-term spatio-temporal features are cut and pooled from the global feature F<sub>k</sub>. 

然后从全局特征F<sub>k</sub>中切出T个短期时空特征并将其合并。

The t-th short-term spatio-temporal feature x<sub>t</sub> is constructed as,

将第t个短期时空特征x<sub>t</sub>构造为，

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/xt.png)

where ltap is local temporal average pool layer in truncated 3D-Densenet, k/T is half of temporal feature interval. 

其中ltap是截断的3D-Densenet中的本地时间平均池层，k/T是时间特征间隔的一半。

In this way, the adjacent ltap windows also overlapping that assure the relevance and completeness of the front and back frame information.

以这种方式，相邻的轻敲窗口也重叠，以确保前后框架信息的相关性和完整性。

After local temporal average pooling, we can get a sequence of short-term features in single modality. 

经过局部时间平均池化后，我们可以在单个模态中获得一系列短期特征。

Multimodal feature sequences are fused into one sequence before input into TCN. 

在输入TCN之前，将多峰特征序列融合为一个序列。

In this paper, all feature sequences of different modality are concated in channel dimension.

在本文中，所有不同形态的特征序列都在信道维度上被概括。

### 3.2 TSENet + TCN for long-term prediction
### 3.2 TSENet + TCN进行长期预测

Based on the short-term spatio-temporal features extracted from all kinds of data modalities (RGB, optic flow, depth, etc.), the long-term temporal features of the whole video is considered to classify the category of the given hand gesture. 

基于从各种数据模式（RGB，光流，深度等）中提取的短期时空特征，可以考虑整个视频的长期时空特征来对给定手势的类别进行分类 。

In this work, a sequence recognition model named TCNs is employed and modified to process the long-term temporal information. 

在这项工作中，采用了一个名为TCN的序列识别模型，并对其进行了修改以处理长期时间信息。

The main characteristics of TCNs are the use of causal convolutions and the mapping of an input sequence to an output sequence of the same length. 

TCN的主要特征是使用因果卷积以及将输入序列映射到相同长度的输出序列。

In addition, accounting for sequences with long history, this model uses dilated convolutions that enable a large receptive field as well as residual connections that allow training deeper networks. 

此外，考虑到历史悠久的序列，此模型使用了扩大的卷积，从而实现了较大的接收场以及残差连接，从而可以训练更深的网络。

Considering that our task is to classify the category of hand gesture videos, the output layer of TCN is further processed by one fully connection layer to obtain a single class label for each gesture sequence. 

考虑到我们的任务是对手势视频的类别进行分类，一个完全连接层会进一步处理TCN的输出层，以获得每个手势序列的单个类别标签。

The structure of the proposed modified version of the TCN model is depicted in Figure 3.

提出的TCN模型的修改版本的结构如图3所示。

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/figure3.png)

**Figure 3: Architecture and architectural elements in a TCN.** 

**图3：TCN中的体系结构和体系结构元素。**

**There is an example that dilated causal convolution with dilation factors d = 1,2,4 and filter size k = 2 in figure.** 

**有一个例子，图中放大系数为d = 1,2,4，滤波器大小k = 2的因果卷积。**

**The receptive field is able to cover all values from the input sequence.**

**接收字段能够覆盖输入序列中的所有值。**

**And adjacent layers are connected by residual block.**

**相邻的层通过残块连接。**

**Before temporal convolution layer, the inputs need to go through the corresponding Temporal Squeeze-andExcitation(TSE) layer to adjust weight of input in temporal domain.**

**在时间卷积层之前，输入需要经过相应的时间挤压和激发（TSE）层，才能在时间域中调整输入的权重。**

The short-term temporal features X = [x1, , xT ] are utilized as the input sequence of the proposed modified TCN with the outputs Y = [y1, , yT ], while the calculation of yt, t < T depends only on X = [x1, , xT ]. 

短期时间特征X = [x1，，xT]用作拟议的改进TCN的输入序列，输出Y = [y1，，yT]，而yt，t <T的计算仅取决于X = [x1，，xT]。

The reason is that the dilated convolutions are calculated as,

原因是膨胀卷积的计算公式如下：

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/yt.png)

where ∗d is the operator for dilated convolutions, d is the dilation factor and h is the filter’s impulse response. 

其中∗ d是扩张卷积的算符，d是扩张因子，h是滤波器的冲激响应。

For a TCN with L layers, the output of the last layer y<sup>L</sup> is used for the sequence classification. 

对于具有L层的TCN，最后一层y <sup>L</sup>的输出用于序列分类。

The class label o_hat attributed to the sequence is found through a fully connected layer with a softmax activation function,

可通过具有softmax激活功能的完全连接的层找到归属于序列的类标签，

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/o_hat.png)

where W<sub>o</sub>, b<sub>o</sub> are trainable parameters.

其中W<sub>o</sub>，b<sub>o</sub>是可训练参数。

It is noting that the short-term spatio-temporal features x<sub>1</sub>, x<sub>T</sub> actuallyhave different contributions to the recognition in the long-term temporal information processing.

值得注意的是，短期时空特征x<sub>1</sub>，x<sub>T</sub>在长期时间信息处理中对识别的贡献是不同的。

For instance, the gesture ”swipe +” in Figure 4(b) contains three paths.

例如，图4（b）中的手势“swipe+”包含三条路径。

The first path is extremely similar to the gesture ”swipe left” (Figure 4(a)) when t < 9.

当t<9时，第一条路径与手势“向左滑动”（图4（a））极其相似。

The same phenomenon occurs between the third path of the gestures ”swipe +” and ”swipe down” (Figure 4(c)) when t > 23.

当t>23时，手势“swipe+”和“swipe down”（图4（c））的第三条路径之间也会出现同样的现象。

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/figure4.png)

**Figure 4: An example sequence from VIVA gesture and its corresponding temporal weghts from TSE-Nets.**

**图4：来自VIVA手势的示例序列及其来自TSE-Nets的相应时间加权。**

In order to assign different temporal weight to X = [x<sub>1</sub>, x<sub>T</sub> ] , a temporal Squeeze-and-Excitation network (TSENet) block is inserted between each temporal convolution layers.

为了给X=[X<sub>1</sub>，X<sub>T</sub>]分配不同的时间权重，在每个时间卷积层之间插入时间挤压和激励网络（TSENet）块。

As shown in Figure ??, the average pooling is applied on the channel dimensions C of X = [x<sub>1</sub>, x<sub>T</sub> ] to squeeze channel-wise information.

如图所示？？，将平均池应用于X=[X<sub>1</sub>，X<sub>T</sub>]的信道维度C，以压缩信道信息。

Such obtained temporal descriptor z = [z<sub>1</sub>,  z<sub>T </sub>] is a T × 1 vector, while the t-th element of z is calculated as,

这样得到的时间描述符z=[z<sub>1</sub>，z<sub>T</sub>]是T×1向量，而z的第T元素计算为，

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/zt.png)

Then another excitation operation is followed to capture the temporal dependencies, i.e. the temporal weights.

然后，接着进行另一个激励操作，以捕获时间相关性，即时间权重。

To fulfil this objective, we opt to employ a simple gating mechanism with the activations:

为了实现这一目标，我们选择使用一个简单的激活门控机制：

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/s.png)

where σ refers to the sigmoid function, δ refers to the ReLU function,

其中，σ是指S状函数，δ是指ReLU函数，

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/w1.png)

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/w2.png)

and r is the size of squeeze channel.

r是挤压槽的尺寸。

The final output of the block is obtained by rescaling the transformation output U with the activations:

块的最终输出是通过使用激活重新缩放变换输出U来获得的：

![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/x~.png)

where ![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/X~.png) and F<sub>scale</sub>(u<sub>t</sub>, s<sub>t</sub>) refers to temporal-wise multiplication between the scalar s<sub>t</sub> and the feature map u<sub>t</sub> ∈ R<sup>T</sup>.

其中![](/media/alex/新加卷/Match/Internet+/Intelligent-disability-assistive-glasses/Documentation/Paper/Short-Term Temporal Convolutional Networks for Dynamic Hand Gesture Recognition/images/X~.png)和F<sub>标度</sub>（u<sub>t</sub>，s<sub>t</sub>）是指标量s<sub>t</sub>和特征映射u<sub>t</sub>之间的时间相乘。


An example of the weights on different TSENet layers is illustrated in Figure5.

图5示出了不同TSENet层上的权重的示例。

It can be seen that the values of the weights changes corresponding to the input gesture sequence as desired.

可以看出，权重值随输入手势序列的需要而变化。

## 4 Experime
## 4 经验

The proposed network architecture is implemented by tensorflow, and trainedusing one NVIDIA Quadro GP100 GPU.

提出的网络结构由tensorflow实现，并用nvidiaquadrogp100gpu进行训练。

Multimodal 3D-DenseNet models havesame structures and are pretrained using RGB and optic flow(if optic flowexisted or can be calculated) data respectively.

多模式3D稠密模型具有结构，并分别使用RGB和光流（如光流存在或可计算）数据进行预训练。

Adam optimizer is used fortraining 3D-DenseNet and the learning rate is initialized to 6.4e−4 and decayedby 10 every 25 epochs.

Adam优化器用于训练3D DenseNet，学习率初始化为6.4e-4，每25个阶段递减10次。

The weight decay is set to 1e − 4.

重量衰减设置为1e-4。

And the dropout rateis set to 0.2.

辍学率设为0.2。

The compression rate and the growth k in the DenseNet block areset as 0.5 and 12, respectively.

DenseNet区块的压缩率和生长k分别为0.5和12。

For the TCN model, we use Adam optimizer fortraining, and the learning rate is initialized to 1e − 4, epsilon is 1e − 8.

对于TCN模型，我们使用Adam优化器进行训练，学习率初始化为1e-4，epsilon为1e-8。

## 4.1 Dataset
## 4.1数据集

In this section, we compare our method with the other state-of-the-art dynamic hand gesture methods.

在本节中，我们将我们的方法与其他最先进的动态手势方法进行比较。

Two publicly available multi-modal dynamic hand gesture datasets (VIVA[14] and NVGesture[5]) are used to evaluate our proposed model in the experiment.

在实验中，我们使用了两个公开的多模态动态手势数据集（VIVA[14]和NVGesture[5]）来评估我们提出的模型。

### VIVA[14]

The VIVA challenges dataset is a multimodal dynamic hand gesture dataset specifically designed with difficult settings of cluttered background,volatile illumination, and frequent occlusion for studying natural human activities in real-world driving settings.

VIVA challenges数据集是一个多模式动态手势数据集，专门设计用于研究真实驾驶环境中的自然人类活动，具有背景杂乱、光照不稳定和频繁遮挡等困难设置。

This dataset was captured using a MicrosoftKinect device, and contains 885 intensity and depth video sequences of 19 different dynamic hand gestures performed by 8 subjects inside a vehicle.

这个数据集是使用MicrosoftKinect设备捕获的，包含885个强度和深度视频序列，其中包括8名受试者在车内执行的19种不同的动态手势。

Figure4 shows some gesture sequences.

图4显示了一些手势序列。

### NVGesture[5]

The NVGesture dataset has been captured with multiplesensors and from multiple viewpoints for studying human-computer interfaces.

为了研究人机界面，采用多传感器多角度采集了NVGesture数据集。

It contains 1532 dynamic hand gestures recorded from 20 subjects inside a casimulator with artificial lighting conditions.

它包含1532个动态手势，这些手势是由20名受试者在人工光照条件下在模拟体内记录下来的。

This dataset includes 25 classes of hand gestures.

这个数据集包括25类手势。

The gestures were recorded with SoftKinetic DS325 device as thRGB-D sensor and DUO-3D for the infrared streams.

用软动态DS325装置作为thRGB-D传感器，用DUO-3D对红外流进行手势记录。

In the experiments, weuse RGB, depth and optical flow modalities, while the optical flow is calculatedfrom the RGB stream using the method presented in [34].

在实验中，我们使用RGB、深度和光流模式，而光流是使用文献[34]中的方法从RGB流计算出来的。

## 4.2 Data Preprocessing
## 4.2数据预处理

In VIVA dataset, data augmentation is comprised of three other operations: reverse ordering of frames, horizontal mirroring, and applying both operationstogether.

在VIVA数据集中，数据扩充由另外三个操作组成：帧的反向排序、水平镜像和同时应用这两个操作。

With these operations we generated additional samples for training.

通过这些操作，我们生成了额外的训练样本。

For example, applying both operations transforms the original gesture ”SwipeLeft” with the right hand to a new gesture ”Swipe Left” with the left hand.

例如，应用这两个操作会将原来右手的手势“swipleft”转换为左手的新手势“swipleft”。

In NVGesture dataset, for special augmentation, videos are resized to have thsmaller video size of 256 pixels, and then randomly cropped with a 224x224 patch.

在NVGesture数据集中，为了进行特殊的增强，视频被调整为256像素的较小视频大小，然后用224x224补丁随机裁剪。

Data normalization is also applied on both datasets, since a fixed dimension of input data is required in the C3D model and TCN model.

由于在C3D模型和TCN模型中需要输入数据的固定维度，因此数据规范化也适用于这两个数据集。

For the videos with different temporal lengths, uniform normalization with temporal upsampling and downsampling is used.

对于不同时间长度的视频，采用时间上采样和下采样的均匀归一化方法。

To compress or extend a given video V with n frames to k frames,

要压缩或扩展给定的视频V（n帧到k帧），请执行以下操作：，

1) If n > k, we split the video V into a k section video set V<sub>S</sub> averagely, where V<sub>S</sub> = [<sub>V1</sub>, V<sub>2</sub>, ..., V<sub>k</sub>].

1） 如果n>k，我们将视频V平均分割成k段视频集V<sub>S</sub>，其中V<sub>S</sub>=[<sub>V1</sub>，V<sub>2</sub>，…，V<sub>k</sub>]。

For each piece in the video set VS, we randomly choose one frame as the representation of the sub-video fragment.

对于视频集VS中的每个片段，我们随机选择一个帧作为子视频片段的表示。

Finally we concatenate all the represent frames and make them as the result of the normalization.

最后，我们将所有表示帧连接起来，并使它们成为规范化的结果。

2) If n < k, we randomly choose k − n frame in the video, then repeat them follow by themselves.

2） 如果n<k，我们随机选择视频中的k-n帧，然后自己重复它们。

In our experiments, the average number of frames k is set as 32 for VIVAdataset and 64 for NVGesture dataset.

在我们的实验中，VIVAdataset的平均帧数k设置为32，NVGesture的平均帧数k设置为64。

Due to the high complexity of 3D convolutional calculating, the spatial size of the inputs is restricted to 112 × 11.

由于3D卷积计算的高度复杂性，输入的空间大小被限制在112 × 11。

### 4.3 Evaluation on VIVA Datase
### 4.3 VIVA Datase评估

Table 2 shows the performance of the dynamic hand gestures tested on the RGB and depth modalities of the VIVA dataset.

表2显示了在VIVA数据集的RGB和深度模式上测试的动态手势的性能。

The compared methods include the hand-crafted approach HOG+HOG2[35], the recurrent CNN-based method(CNN:LRN)[14], the C3D model which were pretrained on Sport-1M dataset, the I3D method[32] that performs very well in action recognition, and the Multimodal Training / Unimodal Testing (MTUT) model[33] which shows promising performance in dynamic hand gesture recognition.

比较的方法包括手工制作的HOG+HOG2[35]、基于CNN的递归方法（CNN:LRN）[14]、在Sport-1M数据集上预训练的C3D模型、在动作识别中表现非常好的I3D方法[32]，以及在动态手势识别中显示出良好性能的多模态训练/单峰测试（MTUT）模型[33]。

All the resultsare reported by averaging the classification accuracies.

所有的结果都是通过平均分类准确度来报告的。

It can be seen that theproposed model achieves the highest accuracy, which is 5.46% higher than thestate-of-the-art method MTUT.

由此可见，所提出的模型达到了最高的精度，比现有的方法MTUT高出5.46%。

This experiment shows that our model is effective to extract both short-term and long-term spatio-temporal information fordynamic hand gesture recognition.

实验表明，该模型能有效地提取动态手势识别的短期和长期时空信息。

To validate the effect of the proposed TSENet layers, the accuracy obtained by vanilla TCN is also shown in Table 2.

为了验证所提议的TSENet层的效果，香草TCN获得的精度也如表2所示。

It can be seen that the presence of the TSENet layers in the TCN can improve the recognition rate by around 0.8%.

可见，在TCN中加入TSENet层可以提高识别率约0.8%。

Three examples of the temporal weights produced by TSENet layers are shown in Figure 6.

由TSENet层产生的时间权重的三个示例如图6所示。

It is interesting to see that the weights in the third layer contain obvious large and small values, which means it does select the important ones from the short-term features.

有趣的是，第三层的权重包含明显的大小值，这意味着它确实从短期特征中选择了重要的权重。

Moreover, if we change the 3D-Dense networks to Res3D which is used for extracting the short-term features.

此外，如果我们将三维密集网络改为Res3D来提取短期特征。

The accuracy will further drop about 4.8%.

准确度将进一步下降约4.8%。

It proves the effectiveness of the structure of the proposed model.

证明了该模型结构的有效性。

Figure 6a shows the confusion matrix as well for the experiment.

图6a显示了实验的混淆矩阵。

It can be seen that the proposed model confused between the Swipe and Scroll gestures performed along the same direction.

可以看出，所提出的模型混淆了沿同一方向执行的滑动和滚动手势。

Many gestures were mis-classified as the Swipe down gesture, the Rotate CW/CCW gestures were difficult for the proposed model.

许多手势被错误地归类为向下滑动手势，而旋转CW/CCW手势对所提出的模型来说是困难的。

In some case, the propose model may have difficulties with distinguishing between the Swipe + and the Swipe X gestures.

在某些情况下，建议的模型可能难以区分Swipe+和Swipe X手势。

### 4.4 EVALUATION ON NVgesture.

The NVGesture dataset, containing RGB, depth and optical flow modalities, is also used to test the proposed model.

利用包含RGB、深度和光流模式的NVGesture数据集对该模型进行了测试。

Table 3 tabulates the results of our method in comparison with the recent state-of-the-art methods: HOG+HOG2[35], improved dense trajectories(iDT)[31], R3DCNN[5], two-stream CNNs[3], and C3Das well as human labeling accuracy.

表3列出了我们的方法与最新技术方法的比较结果：HOG+HOG2[35]、改进的稠密轨迹（iDT）[31]、R3DCNN[5]、双流cnn[3]和C3Das以及人类标记的准确性。

The iDT method is often recognized as thebest performing hand-crafted method.

iDT方法通常被认为是性能最好的手工方法。

However, we observe that similar to the pervious experiments the 3D-CNN-based methods outperform the other handgesture recognition methods, and among them, our method provides the better performance in all the modalities.

然而，我们观察到，与前面的实验类似，基于3D-CNN的方法优于其他手势识别方法，其中，我们的方法在所有模式下都提供了更好的性能。

Nonetheless, compare to the latest method MTUT, our method accuacies are close to the MTUT.

尽管如此，与最新的MTUT方法相比，我们的方法精度接近MTUT。

Our method has the better performance in both of RGB and optical flow modalities, it improve accuracy by 0.73%.

该方法在RGB和光流两种模式下都有较好的性能，准确度提高了0.73%。

But in RGB+Depth modalities and in RGB+Depth+Opt.flow modalities, our method is not performing good enough.

但在RGB+Depth和RGB+Depth+Opt.flow两种模式下，我们的方法都表现得不够好。

This is in part due to the knowledge that gestures in NVGesture are more complex and have more invalid information.

这在一定程度上是因为人们知道nvphere中的手势更复杂，并且包含更多无效信息。

Although through TCN and TSE, our method can key information in the frames and weaken the influence of irrelevant information, the redundant non gesture information, especially in temporal, always affects the final results of the experiment.

虽然通过TCN和TSE，我们的方法可以在帧内提取关键信息，削弱无关信息的影响，但冗余的非手势信息，尤其是时间上的冗余信息，往往会影响实验的最终结果。

## 5 Conclusion
## 5 结论

We developed an effective method for multi-modal (RGB, depth and optic flowdata) dynamic hand gesture recognition with 3D-DenseNets and TCNs.

提出了一种基于三维DenseNets和TCNs的多模态（RGB、深度和光流数据）动态手势识别方法。

And in TCNs, we improved and applied an attention model named SENets to learnand extract deeper temporal features.

在TCNs中，我们改进并应用了一个名为SENets的注意模型来学习和提取更深层次的时间特征。

The experiments show that the proposed model achieved the highest accuracy in VIVA dataset, as well as competitive results in NVGesture dataset.

实验表明，该模型在VIVA数据集上达到了最高的精度，在NVGesture数据集上也取得了很好的效果。

However, our model is still not an end-to-end model and has to be trained step by step.

然而，我们的模型仍然不是一个端到端的模型，必须一步一步地训练。

Meanwhile, NVGesture still have a large room for improvement,we still have a lot of work to enhance the accuracy of the model.

同时，NVGesture还有很大的改进空间，我们在提高模式精度方面还有很多工作要做。

## References
##参考文献

[1] S. S. Rautaray and A. Agrawal, “Vision based hand gesture recognitionfor human computer interaction: a survey,” Artificial intelligence review,vol. 43, no. 1, pp. 1–54, 2015.

[1] S.S.Rautaray和A.Agrawal，“基于视觉的人机交互手势识别：调查”，《人工智能评论》，第43卷，第1期，第1-54页，2015年。



[2] N. C. Camgoz, S. Hadfield, O. Koller, and R. Bowden, “Subunets: End-to-end hand shape and continuous sign language recognition,” in 2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017, pp.3075–3084.

[2] N.C.Camgoz、S.Hadfield、O.Koller和R.Bowden，“Subunets:端到端手形和连续手语识别”，在2017年IEEE国际计算机视觉会议上。IEEE，2017年，第3075-3084页。



[3] K. Simonyan and A. Zisserman, “Two-stream convolutional networks foraction recognition in videos,” in Advances in neural information processin systems, 2014, pp. 568–576.

[3] K.Simonyan和A.Zisserman，“视频中动作识别的两流卷积网络”，《神经信息处理系统进展》，2014年，第568-576页。



[4] N. Neverova, C. Wolf, G. W. Taylor, and F. Nebout, “Multi-scale deeplearning for gesture detection and localization,” in European Conference on Computer Vision. Springer, 2014, pp. 474–490.

[4] N.Neverova，C.Wolf，G.W.Taylor和F.Nebout，“用于手势检测和定位的多尺度深度学习”，在欧洲计算机视觉会议上发表。斯普林格，2014年，第474-490页。



[5] P. Molchanov, X. Yang, S. Gupta, K. Kim, S. Tyree, and J. Kautz, “Onlinedetection and classification of dynamic hand gestures with recurrent 3convolutional neural network,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 4207–4215.

[5] P.Molchanov，X.Yang，S.Gupta，K.Kim，S.Tyree和J.Kautz，“使用递归3进化神经网络对动态手势进行在线检测和分类”，《IEEE计算机视觉和模式识别会议论文集》，2016年，第4207-4215页。



[6] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Denselyconnected convolutional networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 4700–4708.

[6] G.Huang，Z.Liu，L.Van Der Maaten和K.Q.Weinberger，“Densely连接卷积网络”，《计算机视觉和模式识别IEEE会议论文集》，2017年，第4700-4708页。



[7] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv preprint arXiv:1803.01271, 2018.

[7] S.Bai、J.Z.Kolter和V.Koltun，“序列建模用一般卷积和递归网络的经验评估”，arXiv预印本arXiv:1803.012712018。



[8] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition,2018, pp. 7132–7141.

[8] 胡建华，沈立军，孙国强，“挤压与激励网络”，载于美国电气与电子工程师协会计算机视觉与模式识别会议论文集，2018年，7132-7141页。



[9] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification withdeep convolutional neural networks,” in Advances in neural information processing systems, 2012, pp. 1097–1105.

[9] A.Krizhevsky，I.Sutskever和G.E.Hinton，“深度卷积神经网络的图像网络分类”，《神经信息处理系统进展》，2012年，第1097-1105页。



[10] P. Wang, W. Li, S. Liu, Z. Gao, C. Tang, and P. Ogunbona, “Large-scaleisolated gesture recognition using convolutional neural networks,” in 2016 23rd International Conference on Pattern Recognition (ICPR). IEEE,2016, pp. 7–12.13

[10] Wang，W.Li，S.Liu，Z.Gao，C.Tang和P.Ogubona，“使用卷积神经网络的大规模倾斜手势识别”，2016年第23届国际模式识别会议（ICPR）。IEEE，2016年，第7-12.13页



[11] B. Fernando, E. Gavves, J. Oramas, A. Ghodrati, and T. Tuytelaars, “Rankpooling for action recognition,” IEEE transactions on pattern analysis and machine intelligence, vol. 39, no. 4, pp. 773–787, 2016.

[11] B.Fernando，E.Gavves，J.Oramas，A.Ghodrati和T.Tuytelaars，“动作识别的Rankpooling”，模式分析和机器智能的IEEE交易，第39卷，第4期，第773-787页，2016年。



[12] L. Pigou, A. Van Den Oord, S. Dieleman, M. Van Herreweghe, andJ. Dambre, “Beyond temporal pooling: Recurrence and temporal convolutions for gesture recognition in video,” International Journal of Computer Vision, vol. 126, no. 2-4, pp. 430–439, 2018.

[12] L.Pigou，A.Van Den Oord，S.Dieleman，M.Van Herreweghe和J。Dambre，“超越时间池：视频中手势识别的递归和时间卷积”，《国际计算机视觉杂志》，第126卷，第2-4期，第430-439页，2018年。



[13] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, “Learningspatiotemporal features with 3d convolutional networks,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 4489–4497.

[13] D.Tran、L.Bourdev、R.Fergus、L.Torresani和M.Paluri，“利用三维卷积网络学习时空特征”，收录于《IEEE计算机视觉国际会议论文集》，2015年，第4489-4497页。



[14] P. Molchanov, S. Gupta, K. Kim, and J. Kautz, “Hand gesture recognitionwith 3d convolutional neural networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2015, pp1–7.

[14] P.Molchanov、S.Gupta、K.Kim和J.Kautz，“用三维卷积神经网络进行手势识别”，计算机视觉和模式识别研讨会论文集，2015年，第1-7页。



[15] K. Simonyan and A. Zisserman, “Very deep convolutional networks folarge-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.

[15] K.Simonyan和A.Zisserman，“超大规模图像识别的非常深卷积网络”，arXiv预印本arXiv:1409.15562014。



[16] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan,V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” inProceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1–9.

[16] C.Szegedy，W.Liu，Y.Jia，P.Sermanet，S.Reed，D.Anguelov，D.Erhan，V.Vanhoucke，和A.Rabinovich，《卷积的深入》，美国电气与电子工程师协会计算机视觉和模式识别会议论文集，2015年，第1-9页。



[17] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deepnetwork training by reducing internal covariate shift,” arXiv preprint

[17] S.Ioffe和C.Szegedy，“批量规范化：通过减少内部协变量变化加速深度网络培训”，arXiv预印本

arXiv:1502.03167, 2015.

arXiv:1502.031672015年。



[18] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinkingthe inception architecture for computer vision,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 2818–2826.

[18] C.Szegedy、V.Vanhoucke、S.Ioffe、J.Shlens和Z.Wojna，“重新思考计算机视觉的初始体系结构”，《计算机视觉和模式识别IEEE会议论文集》，2016年，第2818-2826页。



[19] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi, “Inception-v4inception-resnet and the impact of residual connections on learning,” inThirty-First AAAI Conference on Artificial Intelligence, 2017.

[19] C.Szegedy，S.Ioffe，V.Vanhoucke和A.A.Alemi，“初始-V4初始-resnet和剩余连接对学习的影响”，第1届人工智能AAAI会议，2017年。



[20] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for imagerecognition,” in Proceedings of the IEEE conference on computer vision an pattern recognition, 2016, pp. 770–778.

[20] K.He，X.Zhang，S.Ren和J.Sun，“图像识别的深度剩余学习”，《IEEE计算机视觉与模式识别会议论文集》，2016年，第770-778页。

[21] Q. Miao, Y. Li, W. Ouyang, Z. Ma, X. Xu, W. Shi, and X. Cao, “Multimodal gesture recognition based on the resc3d network,” in Proceedings on the IEEE International Conference on Computer Vision, 2017, pp. 3047–3055.14

[21]Q.Miao，Y.Li，W.Ouyang，Z.Ma，X.Xu，W.Shi和X.Cao，“基于resc3d网络的多模态手势识别”，《IEEE国际计算机视觉会议论文集》，2017年，第3047-3055.14页

[22] H. J. Escalante, V. Ponce-L´opez, J. Wan, M. A. Riegler, B. Chen,A. Clap´es, S. Escalera, I. Guyon, X. Bar´o, P. Halvorsen et al., “Chalearnjoint contest on multimedia challenges beyond visual analysis: Aoverview,” in 2016 23rd international conference on pattern recognition (ICPR). IEEE, 2016, pp. 67–73.

[22]H.J.Escalante，V.Ponce-L'opez，J.Wan，M.A.Riegler，B.Chen，A.Clap'es，S.Escalera，I.Guyon，X.Bar'o，P.Halvorsen等人，2016年第23届模式识别国际会议，“视觉分析以外的多媒体挑战的Chalearn联合竞赛：Aoverview”（ICPR）。IEEE，2016年，第67-73页。

[23] J. Wan, S. Escalera, G. Anbarjafari, H. Jair Escalante, X. Bar´o, I. Guyon,M. Madadi, J. Allik, J. Gorbova, C. Lin et al., “Results and analysis of chalearn lap multi-modal isolated and continuous gesture recognition, anreal versus fake expressed emotions challenges,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 3189–3197

[23]J.Wan，S.Escalera，G.Anbarjafari，H.Jair Escalante，X.Bar'o，I.Guyon，M.Madadi，J.Allik，J.Gorbova，C.Lin等人，“chalearn-lap多模态孤立和连续手势识别的结果和分析，真实和虚假情绪表达挑战，“在IEEE国际计算机视觉会议记录，2017年，第3189-3197页

[24] T. Zhang, R. Wang, J. Ding, X. Li, and B. Li, “Face recognition based ondensely connected convolutional networks,” in 2018 IEEE Fourth International Conference on Multimedia Big Data (BigMM). IEEE, 2018, pp.1–6.

[24]T.Zhang，R.Wang，J.Ding，X.Li和B.Li，“基于非完全连接卷积网络的人脸识别”，在2018年IEEE第四届多媒体大数据国际会议上。IEEE，2018年，第1-6页。

[25] W. Hao and Z. Zhang, “Spatiotemporal distilled dense-connectivity network for video action recognition,” Pattern Recognition, vol. 92, pp. 13–24,2019.

[25]W.Hao和Z.Zhang，“用于视频动作识别的时空蒸馏密集连接网络”，模式识别，第92卷，第13-242019页。

[26] L. Zhang, G. Zhu, P. Shen, J. Song, S. Afaq Shah, and M. Bennamoun,“Learning spatiotemporal features using 3dcnn and convolutional lstm forgesture recognition,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 3120–3128.

[26]L.Zhang，G.Zhu，P.Shen，J.Song，S.Afaq Shah和M.Bennamoun，“使用3dcnn和卷积lstm伪造识别学习时空特征”，收录于《IEEE国际计算机视觉会议论文集》，2017年，第3120-3128页。

[27] X. Chai, Z. Liu, F. Yin, Z. Liu, and X. Chen, “Two streams recurrentneural networks for large-scale continuous gesture recognition,” in 2016 23rd International Conference on Pattern Recognition (ICPR). IEEE,2016, pp. 31–36.

[27]X.Chai，Z.Liu，F.Yin，Z.Liu和X.Chen，“用于大规模连续手势识别的双流递归神经网络”，2016年第23届国际模式识别会议（ICPR）。IEEE，2016年，第31-36页。

[28] J. Hou, G. Wang, X. Chen, J.-H. Xue, R. Zhu, and H. Yang, “Spatial temporal attention res-tcn for skeleton-based dynamic hand gesture recognition,” in Proceedings of the European Conference on Computer Visio (ECCV), 2018, pp. 0–0.

[28]侯俊杰，王国庆，陈学兴，薛俊海，朱瑞军，杨海阳，“基于骨骼的动态手势识别的时空注意研究”，载《欧洲计算机视觉会议论文集》，2018年，第0-0页。

[29] P. Tsinganos, B. Cornelis, J. Cornelis, B. Jansen, and A. Skodras, “Improved gesture recognition based on semg signals and tcn,” in ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019, pp. 1169–1173.

[29]P.Tsinganos、B.Cornelis、J.Cornelis、B.Jansen和A.Skodras，“基于表面肌电信号和tcn的改进手势识别”，载于ICASSP 2019-2019年IEEE声学、语音和信号处理国际会议（ICASSP）。IEEE，2019年，第1169-1173页。

[30] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances i neural information processing systems, 2017, pp. 5998–6008

[30]A.Vaswani，N.Shazeer，N.Parmar，J.Uszkoreit，L.Jones，A.N.Gomez，L.Kaiser和I.Polosukhin，《注意力是你所需要的一切》，《神经信息处理系统进展》，2017年，第5998-6008页

[31] H. Wang, D. Oneata, J. Verbeek, and C. Schmid, “A robust and efficientvideo representation for action recognition,” International Journal of Computer Vision, vol. 119, no. 3, pp. 219–238, 2016.15

[31]H.Wang，D.Oneata，J.Verbeek和C.Schmid，“动作识别的一种健壮有效的视频表示”，《国际计算机视觉杂志》，第119卷，第3期，第219-238页，2016.15

[32] J. Carreira and A. Zisserman, “Quo vadis, action recognition? a new modeland the kinetics dataset,” in proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 6299–6308.

[32]J.Carreira和A.Zisserman，“vadis，动作识别？一个新的模型和动力学数据集，“在IEEE计算机视觉和模式识别会议记录，2017年，第6299-6308页。

[33] M. Abavisani, H. R. V. Joze, and V. M. Patel, “Improving the performanceof unimodal dynamic hand-gesture recognition with multimodal training,”in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 1165–1174.

[33]M.Abavisani、H.R.V.Joze和V.M.Patel，“通过多模态训练提高单式动态手势识别的性能”，收录于《计算机视觉和模式识别IEEE会议记录》，2019年，第1165-1174页。

[34] G. Farneb¨ack, “Two-frame motion estimation based on polynomial expansion,” in Scandinavian conference on Image analysis. Springer, 2003, pp.363–370.

[34]G.Farneb–ack，“基于多项式展开的两帧运动估计”，在斯堪的纳维亚图像分析会议上发表。斯普林格，2003年，第363-370页。

[35] E. Ohn-Bar and M. M. Trivedi, “Hand gesture recognition in real timefor automotive interfaces: A multimodal vision-based approach and evaluations,” IEEE transactions on intelligent transportation systems, vol. 15,no. 6, pp. 2368–2377, 2014.

[35]E.Ohn Bar和M.M.Trivedi，“汽车接口实时手势识别：基于多模态视觉的方法和评估”，IEEE智能交通系统交易，第15卷，第6期，第2368-2377页，2014年。