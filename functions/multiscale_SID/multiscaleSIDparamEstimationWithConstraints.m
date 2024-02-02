% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function identifies the parameters of the multiscale latent state
% space model from empirical estimates of future-past cross-covariance of 
% latent firing rates and continuous modality H_w, Lambda_0, and mu_all_z_y 
% (defined below).
% The model is described in multiscaleSID.m Matlab function as well as 
% equation set 1 in Ahmadipour et al 2024.
% Inputs:
%       (1) H_w: future-past cross covariance of log firing rates (z_t) and
%           continuous modality (field potentials, y_t) with size 
%           (h_z*n_z+h_y*n_y) by (h_z*n_z+h_y*n_y). See equation 13.
%       (2) Lambda_0: Cov([z_t;y_t]) with size (n_z+n_y) by (n_z+n_y)
%       (3) mu_all_z_y:
%           [mu^{z}; mu^{y}]= E([z^{f}_{t};z^{p}_{t};y^{f}_{t};y^{p}_{t}]) 
%           with size 2*(h_z*n_z+h_y*n_y) by 1. See equation 17.
%       (4) n_z: number of discrete spiking signals N_t 
%           (or number latent log firing rates z_t)
%       (5) n_y: number of continuous field potential signals y_t
%       (6) n_x: dimension of the latent states in multiscale state space 
%           model (equation set 1).
%       (7) h_z: horizon hyper-parameter of Subspace System Identification 
%           algorithm corresponding to discrete spiking observations. 
%           Refer to Ahmadipour et al 2024 for more details.
%       (8) h_y: horizon hyper-parameter of Subspace System Identification
%           algorithm corresponding to spiking observations.
% Outputs:
%       (1) params_N_set: a structure containing the identified multiscale 
%           model parameters N = {A, C_z, C_y, Q, R_y, d_z, d_y}
%       (2) params_P_set: a structure containing an alternative full
%           specification of multiscale model parameters 
%           P = {A, C_z, C_y, G_z, G_y, Lambda_0, d_z, d_y, Sigma_x}.
%           Here G_z = Cov(x_{t+1},z_t), G_y = Cov(x_{t+1},y_t), Sigma_x = Cov(x_t).

function [params_N_set, params_P_set] = multiscaleSIDparamEstimationWithConstraints(H_w, Lambda_0, mu_all_z_y, n_z, n_y, n_x, h_z, h_y)

    H_w = real(H_w);
    Lambda_0 = real(Lambda_0);

    % *************************************************************************
    %               Algorithm 2 - STEP 3 - Ahmadipour et al 2024
    % *************************************************************************
    % Compute the SVD of H_w and keep the n_x largest singular values
    [U, K, V] = svd(H_w);
    U1 = U(:, 1:n_x); K1 = K(1:n_x, 1:n_x); V1 = V(:, 1:n_x);

    % *************************************************************************
    %               Algorithm 2 - STEP 4 - Ahmadipour et al 2024
    % *************************************************************************
    % Compute estimates of the extended observability matrix Obs_w and extended
    % controllability matrix Con_w.
    Obs_w = U1 * sqrt(K1);
    Con_w = sqrt(K1) * V1';

    % *************************************************************************
    %               Algorithm 2 - STEP 5 - Ahmadipour et al 2024
    % *************************************************************************
    % Read C_z and C_y from extended observability matrix Obs_w
    C_z = Obs_w(1:n_z, :);
    C_y = Obs_w(h_z * n_z + 1:h_z * n_z + n_y, :);
    C = [C_z; C_y]; % Concatenating Cz and Cy

    % *************************************************************************
    %               Algorithm 2 - STEP 6 - Ahmadipour et al 2024
    % *************************************************************************
    % Compute A as the least square solution to equation 17. Note that
    % Obs_w=[Obs_z; Obs_y] according to equation 22.
    Obs_z_top = Obs_w([1:h_z * n_z - n_z], :); % Removing the bottom n_z rows of Obs_z
    Obs_z_bottom = Obs_w([n_z + 1:h_z * n_z], :); % Removing the top n_z rows of Obs_z
    Obs_y_top = Obs_w([h_z * n_z + 1:h_z * n_z + h_y * n_y - n_y], :); % Removing the bottom n_y rows of Obs_y
    Obs_y_bottom = Obs_w([h_z * n_z + n_y + 1:h_z * n_z + h_y * n_y], :); % Removing the top n_y rows of Obs_y
    A = [Obs_z_top; Obs_y_top] \ [Obs_z_bottom; Obs_y_bottom];

    % *************************************************************************
    %               Algorithm 2 - STEP 7 - Ahmadipour et al 2024
    % *************************************************************************
    % Read G_z and G_y from extended controllability matrix Con_w
    G_z = Con_w(1:n_x, 1:n_z);
    G_y = Con_w(1:n_x, h_z * n_z + 1:h_z * n_z + n_y);
    G = [G_z, G_y]; % Conacatinating Gz and Gy

    % *************************************************************************
    %               Algorithm 2 - STEP 9 - Ahmadipour et al 2024
    % *************************************************************************
    % Computate valid state noise covariance Q and observation noise covariance
    % R_y by solving a convex optimazation problem
    [Q, R_y, Sigma_x] = findValidNoiseStatistics(A, C, G, Lambda_0, n_z, n_x);

    % *************************************************************************
    %               Algorithm 2 - STEP 10 - Ahmadipour et al 2024
    % *************************************************************************
    % Read d_z and d_y from mu_all_z_y
    d_z = mu_all_z_y(1:n_z);
    d_y = mu_all_z_y(2 * h_z * n_z + 1:2 * h_z * n_z + n_y);

    %% Updating estimates of G_y, G_z and Lambda_0 according to step (vi) of section 2.2.5 in Ahmadipour et al 2024 by setting R_z, R_zy and S to exactly 0.
    R_z = zeros(n_z, n_z);
    S = zeros(n_x, n_z + n_y);
    Lambda_0_update = C * Sigma_x * C' + blkdiag(R_z, R_y);
    G_update = A * Sigma_x * C' + S;
    G_z_update = G_update(:, 1:n_z);
    G_y_update = G_update(:, n_z + 1:end);

    %% Forming the structs containing the multiscale model parameters
    params_P_set = struct('A', A, 'C_z', C_z, 'C_y', C_y, 'G_z', G_z_update, 'G_y', G_y_update, 'Lambda_0', Lambda_0_update, 'd_z', d_z, 'd_y', d_y, 'Sigma_x', Sigma_x);
    params_N_set = struct('A', A, 'C_z', C_z, 'C_y', C_y, 'Q', Q, 'R_y', R_y, 'd_z', d_z, 'd_y', d_y);
end
