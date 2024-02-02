# Multiscale SID algorithm
This repository contains the code for multiscale subspace identification algorithm (multiscale SID algorithm).
For the derivation and results, see the following paper:

Ahmadipour, P., Sani, O. G., Pesaran, B. & Shanechi, M. M. *Multimodal subspace identification for modeling discrete-continuous spiking and field potential population activity.* Journal of Neural Engineering 21, 026001 (2024). https://dx.doi.org/10.1088/1741-2552/ad1053

## Installation guide
In addition to the current repository, also download the following repository: ["CVX: Matlab Software for Disciplined Convex Programming"](http://cvxr.com/cvx/download/). Add all directories from both repositories to the MATLAB path. Then you will be able to run the codes on Matlab.

## Dependencies
You need to install Matlab's Signal Processing Toolbox.

## User guide
The main function is [multiscaleSID.m](./functions/multiscale_SID/multiscaleSID.m), which learns the multiscale model parameters. After learning, model parameters can be passed to  [multiscaleInference.m](./functions/multiscale_inference/multiscaleInference.m) to infer latent states and perform neural prediction.

The script [testScript_multiscaleSID.m](./testScript_multiscaleSID.m) shows a complete example of how the algorithm can be used. This script first loads some simulated multimodal discrete-continuous spiking and field potential population activity and then learns the multiscale model parameters for this data using [multiscaleSID.m](./functions/multiscale_SID/multiscaleSID.m). Afterward, it plots the identified dynamical modes versus the true dynamical modes. Finally, for prediction/estimation of the latent states and neural activity, the learnt parameters and multimodal test time-series are passed to [multiscaleInference.m](./functions/multiscale_inference/multiscaleInference.m). The script then plots predictions of neural activity versus the original neural activity and computes the performance measures. To test on your own data, you just need to provide the multimodal brain network activity time-series as well as a few setting parameters.

## License
Copyright (c) 2024 University of Southern California<br/>
See full notice in [LICENSE.md](./LICENSE.md)<br/>
Parima Ahmadipour, Omid Sani and Maryam M. Shanechi<br/>
Shanechi Lab, University of Southern California<br/>
