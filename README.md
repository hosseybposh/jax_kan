# jax_kan
A Jax implementation of Kolmogorov Arnold Networks.

Original publication: [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)

Original Pytorch implementation: [link](https://github.com/KindXiaoming/pykan)

An efficient pytorch implementation which is the inspiration for this repo: [link](https://github.com/Blealtan/efficient-kan)


# IMPORTANT NOTE
This is a work in progress. It's a port from efficient implementation repo mentioned earlier. Currently everything is tested against the pytorch implementation except for the `update_grid` method. Flax is pretty strict with parameter manipulation outside of computational graph functions so trying to figure out a clean way to do that. Please feel free to contribute.
## Installation
