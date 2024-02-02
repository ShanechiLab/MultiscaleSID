% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function finds the valid latent state noise covariance Q, and
% observation noise covariance R_y by solving the following optimization
% problem (equation 38):
%       min_Sigma_x (||S(Sigma_x)||_F + ||R_z(Sigma_x||_F + 2*||R_{zy}(Sigma_x)||_F)
%       such that Sigma_x, Q(Sigma_x), R_y(Sigma_x) are PSD
% The multiscale latent state space model and its parameters are described 
% in our multiscaleSID.m Matlab function.
% See Ahmadipour et al 2024 for more details.
%
% Inputs: A, C, G, Lambda_0 are parameters of the multiscale model,
% identified from multimodal neural observations in previous steps.
%       (1) A: state transition matrix of the multiscale model (equation
%           1) with size n_x by n_x
%       (2) C: C=[C_z; C_y] with size (n_z+n_y) by n_x, where C_z and C_y 
%           are parameters of the multiscale model
%       (3) G: G= [G_z, G_y] with size n_x by (n_z+n_y) (where G_z and G_y 
%           are parameters of the multiscale model.
%       (4) Lambda_0: Lambda_0=Cov([z_t,y_t] with size (n_y+n_z) by
%           (n_y+n_z)
%       (5) n_z: number of discrete spiking signals N_t (or number latent 
%           log firing rates z_t)
%       (6) n_x: dimension of the latent states
% Outputs:
%       (1) Q: state noise covariance with size n_x by n_x
%       (2) R_y: Continuous observation noise covariance with size n_y by n_y.
%       (3) Sigma_x:latent state covariance with size n_x by n_x

function [Q, R_y, Sigma_x] = findValidNoiseStatistics(A, C, G, Lambda_0, n_z, n_x)
    n_y = size(Lambda_0, 1) - n_z; % number of the continuous field potential signals.
    th_pd = eps;
    cvx_clear;
    % fprintf('Starting CVX...\n');
    % Using CVX to solve the above defined convex optimization problem
    [ ~, ~ ] = cvx_solver; % Select default solver
    cvx_begin quiet
        variable Sigma_x(n_x, n_x) semidefinite;
        R = Lambda_0 - C * Sigma_x * C'; % R=[R_z,R_{zy};R_{zy}',R_{y}];
        Q = Sigma_x - A * Sigma_x * A';
        minimize(norm(G - A * Sigma_x * C', 'fro') + norm(R(1:n_z, 1:n_z), 'fro') + 2 * norm(R(1:n_z, n_z + 1:end), 'fro'));
        (Q - th_pd * eye(n_x)) == semidefinite(n_x);
        (R(n_z + 1:end, n_z + 1:end) - th_pd * eye(n_y)) == semidefinite(n_y);
    cvx_end
    % fprintf('CVX optimization is over\n');
    R_y = R(n_z + 1:end, n_z + 1:end);
end
