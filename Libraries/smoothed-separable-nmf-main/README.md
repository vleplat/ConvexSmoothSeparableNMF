# Smoothed Separable Nonnegative Matrix Factorization

This repository contains the Matlab code and test scripts for Smoothed Separable Nonnegative Matrix Factorization.
It is associated with the article [Smoothed Separable Nonnegative Matrix Factorization](https://arxiv.org/abs/2110.05528).
If this project is of any use in your research, please cite the article.

This is free software, licensed under the [GNU GPL v3](http://www.gnu.org/licenses/gpl.html).

If you have any problem or question when using this code, please [contact me](http://nicolasnadisic.xyz/) and I will be happy to help!


## Problem

Simplex-Structured Matrix Factorization (SSMF) consists in the following optimization problem:
given a data matrix $`X = W H + N \in \mathbb{R}^{m \times n}`$ where $`H \in \mathbb{R}^{r \times n}_{+}`$ is column stochastic and $`N`$ is noise, estimate $`W \in \mathbb{R}^{m \times r}_{+}`$.
Columns of $`X`$ are data points belonging to the convex hull of the columns of $`W`$, called vertices.

Smoothed Separable NMF is a variant of SSMF with the assumption that, for each vertex, there are $`p`$ data points close to this vertex, for a given $`p \in \mathbb{N}`$.


## Algorithms

- `ALLS.m` is the [Algorithm to Learn a Latent Simplex](https://epubs.siam.org/doi/abs/10.1137/1.9781611975994.8), designed by Chiranjib Bhattacharyya and Ravindran Kannan. The code present here is a personnal reimplementation.
- `SVCA.m` is an original contribution. It is a smoothed variant of the algorithm [VCA](https://ieeexplore.ieee.org/abstract/document/1411995).
- `SSPA` is an original contribution. It is a smoothed variant of the algorithm [SPA](https://www.sciencedirect.com/science/article/pii/S0169743901001198).

The folder `utils` contains helper functions that mainly originate from Nicolas Gillis' [repository](https://gitlab.com/ngillis/nmfbook).

## Tests

The test scripts can be run directly to reproduce the experiments featured in the paper.
Files of the form `test_hsu*` and  `test_synth*` correspond respectively to hyperspectral unmixing and synthetic data sets.
