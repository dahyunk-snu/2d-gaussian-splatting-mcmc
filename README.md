# Implementation for 2DGS MCMC and AbsGS

This project extends the [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) by integrating principles from Markov Chain Monte Carlo ([3DGS-MCMC](https://ubc-vision.github.io/3dgs-mcmc/)), as well as [AbsGS](https://github.com/TY424/AbsGS). 

For the MCMC-based relocation mechanism, this implementation uses [Differential Surfel Rasterization MCMC](https://github.com/hwanhuh/diff-surfel-rasterization-MCMC) developed by [@hwanhuh](https://github.com/hwanhuh)(thanks to [@hwanhuh](https://github.com/hwanhuh)).

***The relocation kernel and its behavior strictly follow the original paper and reference implementation, without modification.***
