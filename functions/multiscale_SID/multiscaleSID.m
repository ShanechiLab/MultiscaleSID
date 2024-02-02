% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is the implementation of the unsupervised multiscale SID Algorithm
% in "Multimodal subspace identification for modeling discrete-continuous spiking and field potential population activity" by Ahmadipour et al 2024 (see Algorithm 1 and 2 tables).
% It identifies model parameters of a Multiscale State Space Model in equation 1 from multiscale discrete spiking observations and continuous Gaussian observations.
% The details of the derivation are available in the paper: https://iopscience.iop.org/article/10.1088/1741-2552/ad1053
% 
% The multiscale latent state space model is as follows:
%                           x_{t + 1} = A * x_{t} + q_t,
%                           y_{t} = C_{y} * x_{t} + d_{y} + r_{y,t},
%                           z_{t} = C_{z} * x_{t} + d_{z} + r{z,t},
%                           N_{t} ~ Poisson[exp(z_{t})].
% where x_{t} is the latent state, y_{t} is the continuous field potential activity, 
% z_{t} is the latent log firing rate, and N_{t} is the spiking activity.
% Here, Cov([q_{t};r_{z,t};r{y,t}]) = [Q,S;S',R]
% with S = Cov (q_{t},[r_{z,t}; r_{y,t}]) = 0
% and R = Cov([r_{z,t}; r_{y,t}]) = [R_{z},R_{zy};R_{zy}',R_{y}] =
% [0,0;0,R_{y}].
% The model parameter set that needs to be identified from neural observations
% is N={A, C_z, C_y, Q, R_y, d_z, d_y}. The set P = {A, C_z, C_y, G_z,
% G_y, Lambda_0, d_z, d_y, Sigma_x} is an alternative full specification of
% the multiscale model, where G_z = Cov(x_{t+1},z_t), G_y = Cov(x_{t+1},y_t),
% Lambda_0 =Cov([z_t;y_t]), Sigma_x=Cov(x_t).
% 
% Inputs:
%       (1) data:  a structure containing the time-series of neural observations
%       with the following fields:
%           - N: discrete spiking time series [N_1, N_2,..., N_T] with a size of n_z by T. 
%               n_z is the number of spiking (log firing rate) signals and T is 
%               the total number of time steps.
%           - y: continuous Gaussian times-series (e.g. field potential activity) represented as
%               [y1, NaN, ..., NaN, y_{M+1}, NaN, ..., NaN, y_{2M+1}, ..., y_{T}] 
%               with size of n_y by T.
%               Here, n_y is the number of field signals. 
%               Field signals are observed every M time steps and the missing 
%               observations are marked with NaNs.
%       (2) Settings: a structure with the following fields:
%           - n_x: latent state (x) dimension of multiscale state space model.
%           - h_y: horizon hyper-parameter of Subspace System Identification 
%               algorithm corresponding to field potential (y) observations.
%               Refer to Ahmadipour et al 2024 for more details.
%           - h_z: horizon hyper-parameter of Subspace System Identification
%               algorithm corresponding to spiking observations.
% Outputs:
%       (1) params_N_set: a structure containing the identified multiscale model
%           parameters N = {A, C_z, C_y, Q, R_y, d_z, d_y}
%       (2) params_P_set: a structure containing an alternative full specification of 
%           multiscale model parameters:
%           P = {A, C_z, C_y, G_z, G_y, Lambda_0, d_z, d_y, Sigma_x}.

function [params_N_set, params_P_set] = multiscaleSID(data, settings)

    % *************************************************************************
    %               Algorithm 2 - STEP 1 - Ahmadipour et al 2024
    % *************************************************************************
    % Interpolation of field potentials to recover missing samples
    data.y_interpolated = interpolateSignal(data.y);

    data.y_interpolated = data.y_interpolated(:, 1:size(data.N, 2));
    n_z = size(data.N, 1); n_y = size(data.y_interpolated, 1); n_x = settings.n_x; h_z = settings.h_z; h_y = settings.h_y;

    % Moment computation box in Figure 1(b)
    [Sigma_all_N_y, mu_all_N_y] = multiscaleSIDmomentComputation(data.N, data.y_interpolated, h_z, h_y);

    % *************************************************************************
    %               Algorithm 2 - STEP 2 - Ahmadipour et al 2024
    % *************************************************************************
    % Moment transformation box in Figure 1(b).
    [H_w, Lambda_0, mu_all_z_y] = multiscaleSIDmomentTransformation(Sigma_all_N_y, mu_all_N_y, n_z, n_y, h_z, h_y);

    % *************************************************************************
    %               Algorithm 2 - STEP 3-10 - Ahmadipour et al 2024
    % *************************************************************************
    % Multiscale SID parameter estimation with constraints box in Figure 1(b)
    [params_N_set, params_P_set] = multiscaleSIDparamEstimationWithConstraints(H_w, Lambda_0, mu_all_z_y, n_z, n_y, n_x, h_z, h_y);

end
