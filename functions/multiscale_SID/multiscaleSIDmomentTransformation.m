% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the the future-past cross-covariance of log firing rates (z_t)
% and continuous modality (y_t), given the multiscale model in equation 1.
% The model formulation is also given in multiscaleSID Matlab function.
% Given that log firing rates (z_t) are latent, future-past cross-covariance 
% of log firing rates and continuous modality H_w (see equation 13) is not 
% directly computable from the multimodal observations (N_t and y_t).
% Instead, we compute this cross-covariance H_w by transforming the moments 
% of the discrete-continuous observations computed directly from data 
% (according to the relations in equation 18).
% See multiscaleSIDmomentComputation.m Matlab function for definitions of 
% future (x^{f}) and past (x^{p}) vectors and Ahmadipour et al 2024 for 
% more details.
% Inputs:
%       (1) Sigma_all_N_y: Cov( [N^{f}_{t};N^{p}_{t};y^{f}_{t};y^{p}_{t}]) 
%           with a size of 2*(h_z*n_z+h_y*n_y) by 2*(h_z*n_z+h_y*n_y).
%           Sigma_all_N_y is directly computed from data using our 
%           multiscaleSIDmomentComputation.m Matlab function.
%           Note that Sigma_all_N_y can also be written as 
%           [Sigma^{NN}, Sigma^{Ny}; Sigma^{Ny}', Sigma^{yy}]
%           based on definitions of Sigma^{xx} in equation set 17.
%       (2) mu_all_N_y: E([N^{f}_{t};N^{p}_{t};y^{f}_{t};y^{p}_{t})] 
%           with a size of 2*(h_z*n_z+h_y*n_y) by 1. mu_all_N_y is directly 
%           computed from data using our multiscaleSIDmomentComputation.m 
%           Matlab function. Note that mu_all_N_y can also be written as 
%           [mu^{N};mu^{y}] based on definitions of mu^{x} in equation set 17.
%       (3) n_z: number of discrete spiking signals N_t 
%           (or number latent log firing rates z_t)
%       (4) n_y: number of continuous field potential signals y_t
%       (5) h_z: horizon hyper-parameter of Subspace System Identification 
%           algorithm corresponding to discrete spiking observations. 
%           Refer to Ahmadipour et al 2024 for more details.
%       (6) h_y: horizon hyper-parameter of Subspace System Identification
%           algorithm corresponding to continuous field potential observations.
% Outputs:
%       (1) H_w: future-past cross covariance of log firing rates and
%           continuous modality with size (h_z*n_z+h_y*n_y) by (h_z*n_z+h_y*n_y).
%           H_w=Cov(w^{f}_{t},w^{p}_{t}). w^{f}_{t} (w^{p}_{t}) is formed by
%           stacking the future (past) latent log firing rate vector and the
%           future (past) observed continuous modality. 
%           See equations 13, 14, and 15 for definitions.
%       (2) Lambda_0: Cov([z_t;y_t]) with size (n_z+n_y) by (n_z+n_y)
%       (3) mu_all_z_y: [mu^{z}; mu^{y}]=
%           E([z^{f}_{t};z^{p}_{t};y^{f}_{t};y^{p}_{t}]) with
%           size 2*(h_z*n_z+h_y*n_y) by 1. See equation 17.
% 
% Note: A special case of the multiscale moment transformation for 
% multimodal discrete-continuous observations, is single-scale
% moment transformation for discrete spiking activity only.
% The single-scale moment transformation for spiking activity is derived in 
% "Spectral learning of linear dynamics from generalised-linear
% observations with application to neural population data" by 
% Buesing et al 2012 and implemented here:
% https://bitbucket.org/larsbuesing/ssidforplds/src/master/.

function [H_w, Lambda_0, mu_all_z_y] = multiscaleSIDmomentTransformation(Sigma_all_N_y, mu_all_N_y, n_z, n_y, h_z, h_y)

    eig_min = 1e-5;
    FanoFactor_min = 1.01;
    mu_N = mu_all_N_y(1:2 * h_z * n_z);

    assert(min(mu_N) > 0, 'Some neurons are not firing in the provided time-series. To be able to do moment transformation, those neurons should be removed.');
    
    %% Transformation of moments: Transforming spiking activity moments to log firing rate moments. See derivation of moment transformation for Spiking activity in Buesing et al 2012.
    Sigma_NN_dim = 2 * h_z * n_z; % Dimension of Sigma_NN as defined in equation 17 in Ahmadipour et al 2024
    Sigma_NN = Sigma_all_N_y(1:Sigma_NN_dim, 1:Sigma_NN_dim);
    mu_N = mu_all_N_y(1:Sigma_NN_dim);

    FanoFactor = diag(Sigma_NN) ./ mu_N; % Computing fano factor for each spiking signal
    scale = nan(Sigma_NN_dim, 1);

    for i = 1:Sigma_NN_dim

        if FanoFactor(i) < FanoFactor_min
            scale(i, 1) = sqrt(FanoFactor_min ./ FanoFactor(i));
        else
            scale(i, 1) = 1;
        end

    end

    Sigma_NN_scaled = diag(scale) * Sigma_NN * diag(scale); % Scaling Sigma_NN such that the Fano factor of neurons becomes at least FanoFactor_min.

    % Transforming momemnts of spiking activity to moments of firing rate based on equation 18 in Ahmadipour et al 2024.
    Sigma_zz = log(Sigma_NN_scaled + mu_N * mu_N' - diag(mu_N)) - log(mu_N * mu_N'); % Sigma^{zz}= Cov([z^{f}_{t};z^{p}_{t}])
    mu_z = 2 * log(mu_N) -1/2 * log(diag(Sigma_NN_scaled) + (mu_N .* mu_N) - mu_N);

    %% Transformation of moments: Transforming cross covariance of field potentials and spiking activity to cross covariance of field potentials and log firing rates . See Ahmadipour et al 2024 for derivation.
    Sigma_Ny = Sigma_all_N_y(1:Sigma_NN_dim, Sigma_NN_dim + 1:end);
    Sigma_Ny_scaled = Sigma_Ny .* repmat(scale, 1, 2 * h_y * n_y);
    mu_N_rep = repmat(mu_N, 1, 2 * h_y * n_y);
    Sigma_zy = Sigma_Ny_scaled ./ mu_N_rep; % Sigma^{zy}= Cov([z^{f}_{t};z^{p}_{t}], [y^{f}_{t};y^{p}_{t}] ), see equations 17 and 18.
    
    %% Field potential activity moments
    Sigma_yy = Sigma_all_N_y(Sigma_NN_dim + 1:end, Sigma_NN_dim + 1:end); % Sigma^{yy}= Cov([y^{f}_{t};y^{p}_{t}]), see equations 17 and 18.
    mu_y = mu_all_N_y(Sigma_NN_dim + 1:end);

    %%
    Sigma_all_z_y = [Sigma_zz, Sigma_zy; Sigma_zy', Sigma_yy]; % Sigma_all_z_y= Cov([z^{f}_{t};z^{p}_{t};y^{f}_{t};y^{p}_{t}])
    mu_all_z_y = [mu_z; mu_y];
    %%
    [V, D] = eig(Sigma_all_z_y);

    if min(diag(D)) < eig_min
        D = diag(max(diag(D), eig_min)); % Making sure the minimum eigenvalue of the covariance matrix Sigma_all_z_y is larger than eig_min.
        Sigma_all_z_y = V * D * V';
    end

    Sigma_all_z_y = (Sigma_all_z_y + Sigma_all_z_y') / 2; % Making sure the covariance matrix Sigma_all_z_y is symmetric.
    
    %% Constructing Lambda_0 from Sigma_all_z_y (see Appendix C in Ahmadipour et al 2024)
    Sigma_zz_dim = 2 * h_z * n_z;
    Lambda_0_zz = Sigma_all_z_y(1:n_z, 1:n_z);
    Lambda_0_zy = Sigma_all_z_y(1:n_z, Sigma_zz_dim + 1:Sigma_zz_dim + n_y);
    Lambda_0_yz = Sigma_all_z_y(Sigma_zz_dim + 1:Sigma_zz_dim + n_y, 1:n_z);
    Lambda_0_yy = Sigma_all_z_y(Sigma_zz_dim + 1:Sigma_zz_dim + n_y, Sigma_zz_dim + 1:Sigma_zz_dim + n_y);
    Lambda_0 = [Lambda_0_zz, Lambda_0_zy; Lambda_0_yz, Lambda_0_yy];
    
    %% Constructing H_w from Sigma_all_z_y (see Appendix C in Ahmadipour et al 2024)
    Sigma_yy_dim = 2 * h_y * n_y; % Dimension of Sigma_yy as defined in equation 17.
    H_w_zz = Sigma_all_z_y(1:Sigma_zz_dim / 2, Sigma_zz_dim / 2 + 1:Sigma_zz_dim);
    H_w_zy = Sigma_all_z_y(1:Sigma_zz_dim / 2, Sigma_zz_dim + Sigma_yy_dim / 2 + 1:end);
    H_w_yz = Sigma_all_z_y(Sigma_zz_dim + 1:Sigma_zz_dim + Sigma_yy_dim / 2, Sigma_zz_dim / 2 + 1:Sigma_zz_dim);
    H_w_yy = Sigma_all_z_y(Sigma_zz_dim + 1:Sigma_zz_dim + Sigma_yy_dim / 2, Sigma_zz_dim + Sigma_yy_dim / 2 + 1:end);
    H_w = [H_w_zz, H_w_zy; H_w_yz, H_w_yy]; %H_w=Cov(w^{f}_{t},w^{p}_{t})

end
