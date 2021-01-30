# Aladin Light-Weight-Model-Zoo 
压缩神经网络技术汇总，目前的压缩技术大致可分为以下五类：
- 剪枝
  - 结构化剪枝
  - 非结构化剪枝
- 量化
  - 二值化
  - 固定bit
  - 混合精度
- 网络结构再设计/搜索
  - 轻量化设计
  - 神经网络结构搜索
- 低秩分解
- 网络蒸馏

## 剪枝
- 2019-ICML-[Collaborative Channel Pruning for Deep Networks](http://proceedings.mlr.press/v97/peng19c.html)
- 2019-ICLR-[Dynamic Channel Pruning: Feature Boosting and Suppression](https://openreview.net/forum?id=BJxh2j0qYm)
- 2019-AAAIo-[A layer decomposition-recomposition framework for neuron pruning towards accurate lightweight networks](https://arxiv.org/abs/1812.06611)
- 2019-NIPS-[AutoPrune: Automatic Network Pruning by Regularizing Auxiliary Parameters](https://nips.cc/Conferences/2019/Schedule?showEvent=14303)
- 2019-CVPR-[Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](https://arxiv.org/abs/1904.03837) [[Code](https://github.com/ShawnDing1994/Centripetal-SGD)]
- 2019-ICLR-[Double Viterbi: Weight Encoding for High Compression Ratio and Fast On-Chip Reconstruction for Deep Neural Network](https://openreview.net/forum?id=HkfYOoCcYX)
- 2019-CVPR-[Exploiting Kernel Sparsity and Entropy for Interpretable CNN Compression](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Exploiting_Kernel_Sparsity_and_Entropy_for_Interpretable_CNN_Compression_CVPR_2019_paper.html)
- 2019-CVPRo-[Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250) [[Code](https://github.com/he-y/filter-pruning-geometric-median)]
- 2019-NIPS-[Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1909.08174) [[Pytorch Code](https://github.com/youzhonghui/gate-decorator-pruning)]
- 2019-NIPS-[Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://arxiv.org/abs/1909.12778)
- 2019-CVPR-[Importance Estimation for Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html) [[Code](https://github.com/NVlabs/Taylor_pruning)]
- 2019-ICCV-[MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/abs/1903.10258) 
- 2019-NIPS-[Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717) [[Code](https://github.com/D-X-Y/TAS)]
- 2019-IJCAI-[Play and Prune: Adaptive Filter Pruning for Deep Model Compression](https://arxiv.org/abs/1905.04446)
- 2019-ICLR-[SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://openreview.net/forum?id=B1VZqjAcYX)
- 2019-CVPR-[Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://arxiv.org/abs/1903.09291)
- 2019-ICML-[Training CNNs with Selective Allocation of Channels](http://proceedings.mlr.press/v97/jeong19c.html)
- 2019-CVPR-[Variational Convolutional Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.html)


## 量化
- 2019-NIPS-[Focused Quantization for Sparse CNNs](https://nips.cc/Conferences/2019/Schedule?showEvent=13686)
- 2019-CVPR-[Accelerating Convolutional Neural Networks via Activation Map Compression](http://openaccess.thecvf.com/content_CVPR_2019/html/Georgiadis_Accelerating_Convolutional_Neural_Networks_via_Activation_Map_Compression_CVPR_2019_paper.html)
- 2019-ICLR-[Learning Recurrent Binary/Ternary Weights](https://openreview.net/forum?id=HkNGYjR9FX)
- 2019-ICLR-[Minimal Random Code Learning: Getting Bits Back from Compressed Model Parameters](https://openreview.net/forum?id=r1f0YiCctm)
- 2019-NIPS-[A Mean Field Theory of Quantized Deep Networks: The Quantization-Depth Trade-Off](https://papers.nips.cc/paper/8926-a-mean-field-theory-of-quantized-deep-networks-the-quantization-depth-trade-off)
- 2019-ICLR-[Analysis of Quantized Models](https://openreview.net/forum?id=ryM_IoAqYX)
- 2019-ICLR-[Defensive Quantization: When Efficiency Meets Robustness](https://arxiv.org/abs/1904.08444)
- 2019-NIPS-[Double Quantization for Communication-Efficient Distributed Optimization](https://nips.cc/Conferences/2019/Schedule?showEvent=13598)
- 2019-NIPS-[Random Projections with Asymmetric Quantization](https://papers.nips.cc/paper/9268-random-projections-with-asymmetric-quantization)
- 2019-ICLR-[Relaxed Quantization for Discretized Neural Networks](https://openreview.net/forum?id=HkxjYoCqKX)

## 网络结构再设计/搜索
**轻量化设计**
- 2019-CVPR-[Fast neural architecture search of compact semantic segmentation models via auxiliary cells](https://arxiv.org/abs/1810.10804)
- 2019-CVPR-[All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification](https://arxiv.org/abs/1903.05285)
- 2019-AAAI-[CircConv: A Structured Convolution with Low Complexity](https://arxiv.org/abs/1902.11268)
- 2019-CVPR-[DupNet: Towards Very Tiny Quantized CNN With Improved Accuracy for Face Detection](http://openaccess.thecvf.com/content_CVPRW_2019/html/EVW/Gao_DupNet_Towards_Very_Tiny_Quantized_CNN_With_Improved_Accuracy_for_CVPRW_2019_paper.html)
- 2019-CVPR-[HetConv Heterogeneous Kernel-Based Convolutions for Deep CNNs](https://arxiv.org/abs/1903.04120)
- 2019-ICML-[LegoNet: Efficient Convolutional Neural Networks with Lego Filters](http://proceedings.mlr.press/v97/yang19c.html) [[Code](https://github.com/zhaohui-yang/LegoNet_pytorch)]
- 2019-CVPR-[MBS: Macroblock Scaling for CNN Model Reduction](http://openaccess.thecvf.com/content_CVPR_2019/html/Lin_MBS_Macroblock_Scaling_for_CNN_Model_Reduction_CVPR_2019_paper.html)
- 2020-CVPR-[GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907) [[Code](https://github.com/huawei-noah/ghostnet)]


**神经网络结构搜索**
- 2019-CVPR-[A Neurobiological Evaluation Metric for Neural Network Model Search](https://arxiv.org/pdf/1805.10726.pdf)
- 2019-CVPR-[Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/pdf/1901.02985.pdf) 
- 2019-ICLR-[Efficient Multi-Objective Neural Architecture Search via Lamarckian Evolution](https://arxiv.org/abs/1804.09081)
- 2019-CVPR-[Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells](https://arxiv.org/abs/1810.10804)
- 2019-ICLR-[Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/abs/1810.05749)
- 2019-CVPR-[Haq: Hardware-aware automated quantization with mixed precision](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html)
- 2019-CVPR-[MFAS: Multimodal Fusion Architecture Search](https://arxiv.org/pdf/1903.06496.pdf)
- 2019-CVPR-[MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) 
- 2019-ICML-[NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635) [[Code](https://github.com/google-research/nasbench)]
- 2019-ICCV-[Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) [[Code](https://github.com/chenxin061/pdarts)]
- 2019-ICLR-[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332) [[Code](https://github.com/MIT-HAN-LAB/ProxylessNAS)]


## 低秩分解
- 2019-CVPR-[Compressing Convolutional Neural Networks via Factorized Convolutional Filters](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Compressing_Convolutional_Neural_Networks_via_Factorized_Convolutional_Filters_CVPR_2019_paper.html)
- 2019-CVPR-[Efficient Neural Network Compression](https://arxiv.org/abs/1811.12781) [[Code](https://github.com/Hyeji-Kim/ENC)]
- 2019-[BasisConv: A method for compressed representation and learning in CNNs](https://arxiv.org/abs/1906.04509)
- 2019.09-[Training convolutional neural networks with cheap convolutions and online distillation](https://arxiv.org/abs/1909.13063)

## 网络蒸馏
