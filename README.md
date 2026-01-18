# Scalable Bayesian Inference via Stochastic Gradient Langevin Dynamics

This project investigates scalable Bayesian inference using Stochastic Gradient Langevin Dynamics (SGLD). 
The aim is to compare SGLD with classical MCMC methods and evaluate its effectiveness for uncertainty-aware learning on larger datasets.
This mini-project was completed during the Michaelmas term as part of the Machine Learning module in my MSc programme.

***
## 1 – Sampling Algorithms

In this project, we compare three sampling algorithms: **Random-Walk Metropolis (RWM)**,
the **Metropolis–Adjusted Langevin Algorithm (MALA)**, and **Stochastic Gradient
Langevin Dynamics (SGLD)**.

The key distinction is computational. While RWM and MALA require full-data
likelihood or gradient evaluations at each iteration, SGLD replaces the exact
posterior gradient with an unbiased minibatch estimate. As a result, each update
step has lower computational cost and scales more effectively to large datasets.
Moreover, under an asymptotically vanishing step-size schedule, SGLD is guaranteed
to converge to the true posterior distribution.

The SGLD update is given by:
```math
\theta_{t+1}
=
\theta_t
+
\frac{\epsilon_t}{2}
\left(
\nabla_{\theta} \log p(\theta_t)
+
\frac{N}{n}
\sum_{i=1}^{n}
\nabla_{\theta} \log p(x_i \mid \theta_t)
\right)
+
\eta_t
```

We applied these algorithms to Bayesian logistic regression on the Australian
Credit Approval dataset. Despite their differing computational properties, all
three methods produced closely aligned posterior distributions, validating the
use of SGLD as a scalable approximation to classical MCMC in this setting.

<img width="3000" height="1000" alt="parameter_dist" src="https://github.com/user-attachments/assets/bb563c83-6a26-44f0-ba76-4f079463d5e4" />

***
## 2–Tuning Parameters

When applying SGLD to a dataset, we must make sure to tune the step-size schedule correctly. We use a decaying step size and choose the initial value $\epsilon_0$ so that stochastic minibatch noise dominates initially, while the injected Langevin noise becomes dominant later. To achieve this behaviour, we perform a simple parameter search over $\epsilon_0$. Samples are retained only after this transition has occurred. The initial phase,during which the algorithm behaves similarly to stochastic optimisation before entering a sampling regime, is treated as burn-in and discarded in subsequent analysis.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/e6edcb5e-1095-4036-9f7a-4c70dac44165"
    width="700"
    height="500"
    alt="sgld_burnin_noise_variance"
  />
</p>


***
## 3–Uncertainity 

We then examined predictive uncertainty, which is a central advantage of Bayesian sampling methods over optimisation based approaches. Using the Australian Credit Approval
dataset, we observe that predictive uncertainty varies substantially across test points, even when mean predicted probabilities are similar. For some inputs, predictions are highly stable across posterior samples, resulting in narrow credible intervals. For others, small changes in the model parameters lead to large variations in predicted probability, producing wide credible intervals. This variability reflects uncertainty in the model parameters from the data and cannot be captured by point estimates alone. In contrast, standard optimisation-based methods, which rely on a single parameter estimate, cannot express this distinction between stable and unstable predictions.


<p align="center">
<img width="1400" height="800" alt="post_pred_prob" src="https://github.com/user-attachments/assets/257c792b-d715-4985-b0d2-026e5239ab3e" />
</p>


***
## 4–cSGLD 

In our previous logistic regression example, the posterior distribution was largely unimodal and well behaved. In contrast, the posterior distribution over a neural network’s parameters (weights) is typically high-dimensional and highly multimodal. When using a standard decaying step size in SGLD, the sampler rapidly converges to a single mode and fails to adequately explore the posterior distribution. To mitagate this issue we employed a**cyclical step-size schedule**. This schedule preserves asymptotic correctness by ensuring a vanishing step size, while also mitigating collapse to a single mode by periodically reintroducing larger step sizes.

<p align="center">
<img width="2895" height="1769" alt="stepsize_decay" src="https://github.com/user-attachments/assets/a546be61-09fa-4d11-bba2-08772f614b7e" />
</p>

***
## 5–Bayesian Neural Network
