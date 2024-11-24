# IFL-GAN: Improved Federated Learning Generative Adversarial Network With Maximum Mean Discrepancy Model Aggregation

This readme file is an outcome of the [CENG501 (Fall 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Fall 2024) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction
Generative Adversarial Networks (GANs) are powerful generative models that simulate data distributions by training a generator-discriminator pair in a competitive setting. However, traditional GANs often require centralized and independent, identically distributed (i.i.d.) training data, which is impractical in many real-world scenarios. Data is often distributed among multiple clients and is non-i.i.d., posing challenges for effective GAN training.

Federated Learning (FL) has been applied to address this issue by training GANs in a decentralized manner. Existing approaches, like Federated Learning GAN (FL-GAN), aggregate updates using Federated Averaging (FedAvg), but this method suffers in non-i.i.d. cases, leading to poor convergence and low-quality outputs.

The paper proposes Improved Federated Learning GAN (IFL-GAN), which uses Maximum Mean Discrepancy (MMD) to aggregate updates, assigning weights based on local GAN convergence status. This approach achieves faster training convergence and higher-quality outputs on non-i.i.d. datasets.

## 1.1 Paper Summary
The contributions of the paper can be summarized as:

- A novel framework called IFL-GAN that replaces federating average strategy (FedAvg) with MMD for aggregating local GAN updates, improving performance in non-i.i.d. data settings.
- Theoretical and empirical comparisons of MMD and FedAvg aggregation methods.
- Extensive experimental validation showing superior performance in terms of data diversity and quality across MNIST, CIFAR10, and SVHN datasets.
The IFL-GAN system model is depicted in figure 1.:

<p align="center">
  <img src="figures/overall_diagram_of_rde.png" alt="Cropped regions selected from an image available in COCO dataset with BYOL." style="width: 70%;"><br>
  <em>Figure 1: Overall diagram of RDE</em>
</p>
 

  
### Highlights:
MMD Aggregation: Calculates the supremum difference between source and target distributions to assign variable weights to local GAN updates.
Better Convergence: MMD prioritizes updates from less-converged local GANs, avoiding pitfalls of uniformly weighting updates in FedAvg.
Robust Performance: IFL-GAN produces higher-quality samples and demonstrates faster convergence even on imbalanced or non-i.i.d. datasets.

# 2. Methodology
## 2.1. Key Techniques
## 2.1.1. Training IFL-GAN With MMD
Each local GAN trains on client-specific data and generates a local update.
MMD measures the distributional difference between real data and generated data for each local GAN, normalized via a Softmax function.
The global generator aggregates updates using MMD-weighted parameters:


