# IFL-GAN: Improved Federated Learning Generative Adversarial Network With Maximum Mean Discrepancy Model Aggregation

This readme file is an outcome of the [CENG501 (Fall 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Fall 2024) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction
Generative Adversarial Networks (GANs) are powerful generative models that simulate data distributions by training a generator-discriminator pair in a competitive setting. However, traditional GANs often require centralized and independent, identically distributed (i.i.d.) training data, which is impractical in many real-world scenarios. Data is often distributed among multiple clients and is non-i.i.d., posing challenges for effective GAN training.

Federated Learning (FL) has been applied to address this issue by training GANs in a decentralized manner. Existing approaches, like Federated Learning GAN (FL-GAN), aggregate updates using Federated Averaging (FedAvg), but this method suffers in non-i.i.d. cases, leading to poor convergence and low-quality outputs.

This paper proposes Improved Federated Learning GAN (IFL-GAN), which uses Maximum Mean Discrepancy (MMD) to aggregate updates, assigning weights based on local GAN convergence status. This approach achieves faster training convergence and higher-quality outputs on non-i.i.d. datasets.

## 1.1 Paper Summary
The contributions of the paper can be summarized as:

- A novel framework called IFL-GAN that replaces federating average strategy (FedAvg) with MMD for aggregating local GAN updates, improving performance in non-i.i.d. data settings.
- Theoretical and empirical comparisons of MMD and FedAvg aggregation methods.
- Extensive experimental validation showing superior performance in terms of data diversity and quality across MNIST, CIFAR10, and SVHN datasets.
The IFL-GAN system model is depicted in Figure 1:

<p align="center">
  <img src="figures/system model.png" style="width: 70%;"><br>
  <em>Figure 1: IFL-GAN system model</em>
</p>
 

  
### Highlights:
MMD Aggregation: Calculates the supremum difference between source and target distributions to assign variable weights to local GAN updates.
Better Convergence: MMD prioritizes updates from less-converged local GANs, avoiding pitfalls of uniformly weighting updates in FedAvg.
Robust Performance: IFL-GAN produces higher-quality samples and demonstrates faster convergence even on imbalanced or non-i.i.d. datasets.

# 2. Methodology
## 2.1. Key Concepts

### Generative Adversarial Networks
Generative Adversarial Networks (GANs) consist of two neural networks: a Generator (G) and a Discriminator (D), trained in an adversarial manner. The generator creates synthetic data, while the discriminator attempts to distinguish between real and generated data. The GAN objective function is:

$$min_G  max_D V(G, D) = E_{x ~~ p_r(x)} [log D(x)] + E_{z ~~ p_z(z)} [log (1 - D(G(z)))]$$

Where:

$D(x)$: Probability that x is a real data sample.

$G(z)$: Synthetic data generated from input noise z.

$p_r(x)$: Real data distribution.

$p_z(z)$: easy-to-sample Noise distribution (e.g., Gaussian or uniform).

### Federated Learning for GANs
Federated Learning (FL) enables decentralized training across multiple clients without transferring raw data, preserving privacy. Instead, clients train local models and share their updates with a central server. However, training GANs in a federated setting introduces challenges such as:

- Non-i.i.d. Data: Data across clients often have unique distributions.

- Stability Issues: The adversarial nature of GANs makes distributed training more complex.
FL-GAN extends federated learning to GANs by aggregating updates from multiple local GANs to train a global GAN.

### Federated Learning GAN (FL-GAN)
FL-GAN uses a Federated Averaging (FedAvg) strategy to aggregate updates from local GANs into a global GAN model. Each client maintains a local GAN, with a generator (G_i) and a discriminator (D_i).

FL-GAN Objective Function:
The FL-GAN training objective is:

$min_G max_D V(G, D) = (1/K) * Σ_{i=1}^{K} [E_{x ~ p_{r_i}(x)} log D_i(x) + E_{z ~ p_z(z)} log (1 - D_i(G_i(z)))]$

Where:

$K$: Number of clients.

$G_i, D_i$: Local generator and discriminator for client i.

$p_{r_i}(x)$: Real data distribution for client i.

$G_i(z)$: Data generated by the local generator for client i.

Challenges in FL-GAN
Despite its utility, FL-GAN faces several challenges:

Non-i.i.d. Data: FedAvg assumes equal contribution from all clients, which is ineffective when data distributions vary.

Convergence Instability: Aggregating updates from clients with different convergence statuses can destabilize training.

Low Diversity: Uniform weighting may reduce diversity and fidelity in generated data.

These limitations motivate the development of Improved Federated Learning GAN (IFL-GAN), which uses Maximum Mean Discrepancy (MMD) for weighted aggregation to address these issues effectively.

## 2.2. IFL-GAN

<p align="center">
  <img src="figures/pseudocode for the algorithm.png" style="width: 70%;"><br>
  <em>Figure 1: IFL-GAN system model</em>
</p>




