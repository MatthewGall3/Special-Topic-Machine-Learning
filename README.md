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
2–Tuning Parameters



***
3–Uncertainity Graph


***
4–cSGLD 


***
5–Bayesian Neural Network
