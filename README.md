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
**θ_{t+1} = θ_t + (ε_t / 2)(∇θ log p(θ_t) + (N / n) ∑_{i=1}^n ∇θ log p(x_i | θ_t)) + η_t
**

We applied these algorithms to Bayesian logistic regression on the Australian
Credit Approval dataset. Despite their differing computational properties, all
three methods produced closely aligned posterior distributions, validating the
use of SGLD as a scalable approximation to classical MCMC in this setting.

***
2–Tuning Parameters



***
3–Uncertainity Graph


***
4–cSGLD 


***
5–Bayesian Neural Network
