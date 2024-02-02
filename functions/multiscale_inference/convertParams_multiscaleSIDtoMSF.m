% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function converts the parameter set N= {A,C_y,C_z,Q,R_y d_z,d_y}
% learnt by our "multiscaleSID.m" Matlab function to the parameter set that
% the multiscale filter (MSF) uses {A,C,Theta,Q,R,Bias}. The MSF is implemented in "Decoder.m"
% Matlab function by Abbaspourazad et al (see Abbaspourazad et al 2019, Hsieh et al 2019). 
% Note that the multiscale models assumed by "multiscaleSID.m" and "Decoder.m" 
% functions are exactly the same and just use different names for
% their parameters (see below).
% The model structure that multiscale SID assumes is (see Ahmadipour et al 2024):
%                           x_{t + 1} = A * x_{t} + q_t,
%                           y_{t} = C_{y} * x_{t} + d_{y} + r_{y,t},
%                           z_{t} = C_{z} * x_{t} + d_{z} + r{z,t},
%                           N_{t} ~ Poisson[exp(z_{t})].
% Where x_{t} is the latent state, y_{t} is the continuous field potential 
% activity, z_{t} is the latent log firing rate, and N_{t} is the spiking activity.
% Here, Cov([q_{t};r_{z,t};r{y,t}]) = [Q,S;S',R]
% with S = Cov (q_{t},[r_{z,t}; r_{y,t}]) = 0
% and R= Cov([r_{z,t}; r_{y,t}]) = [R_{z},R_{zy};R_{zy}',R_{y}] =
% [0,0;0,R_{y}].
% The model structure that multiscale filter that (MSF) assumes is:
%                       x_{t + 1} = A * x_{t} + q_t; Cov(q_t) = Q
%                       y_{t} = C * x_{t} + r_t + Bias; Cov(r_t) = R
%                       p(N_{t}|x_{t}) = (lambda(x_{t})* Delta) ^ (N_{t}) * exp( -lambda(x_{t}) * Delta ); lambda(x_{t}) = exp(alpha'*x_{t} + beta), Delta: time-step
%      Or alternatively p(N_{t}|x_{t}) ~ Poisson[lambda(x_{t})* Delta]
% Theta is then defined as parameters of spike modulation -> [beta_c;alpha_c]in each column of Theta for every neuron.
% Input:
%       (1) params_multiscaleSID: A structure with fields A,C_y,C_z,Q,R_y d_z and d_y.
%       (2) Delta: time step size in seconds
% Output:
%       (1) params_MSF: A structure with fields A,C,Theta,Q,R and Bias.

function params_MSF = convertParams_multiscaleSIDtoMSF(params_multiscaleSID, Delta)
    % params_multiscaleSID: N_set={A,C_y,C_z,Q,R_y,d_z,d_y}
    n_z = size(params_multiscaleSID.C_z, 1);
    n_x = size(params_multiscaleSID.A, 1);

    params_MSF.A = params_multiscaleSID.A;
    params_MSF.C = params_multiscaleSID.C_y;
    params_MSF.Q = params_multiscaleSID.Q;

    if isfield(params_multiscaleSID, 'R_y')
        params_MSF.R = params_multiscaleSID.R_y;
    elseif isfield(params_multiscaleSID, 'R')
        params_MSF.R = params_multiscaleSID.R(n_z + 1:end, sDim + 1:end);
    end

    params_MSF.Bias = params_multiscaleSID.d_y;
    params_MSF.Theta = nan(n_x + 1, n_z);
    params_MSF.Theta(1, :) = params_multiscaleSID.d_z' - log(Delta);
    params_MSF.Theta(2:end, :) = params_multiscaleSID.C_z';
end
