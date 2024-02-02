% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the covariance and mean of [N^{f}_{t};N^{p}_{t};y^{f}_{t};y^{p}_{t})]
% from spiking and field potential observations (N_t and y_t).
% Here y^{f}_{t} represents the future field potential activity vector and is formed by stacking
% the time lagged observations within the horizon h_y, i.e.
% y^{f}=[y_{t};y_{t+1};...;y_{t+h_y-1}]. y^{p}_{t} represents the past
% field potential activity vector and is defined as y^{p}=[y_{t-1};y_{t-1};...;y_{t-h_y}].
% Similarly by stacking time-lagged spiking activity over the horizon of h_z, 
% the future and past spiking activity vectors are formed and denoted by N^{f}_{t} and N^{p}_{t}.
% See equations 7 and 8 in Ahmadipour et al 2024.
% 
% Inputs:
%       (1) N_obs: time-series of the discrete spiking activity with size n_z by T. 
%               Here n_z is the number of spiking signals and T is the number of time steps.
%       (2) y_obs: time-series of the continuous field potential activity with
%           size n_y by T. Here n_y is the number of field potential signals. We
%           assume identical sampling rates for y_obs and N_obs with no NaN 
%           values in the time-series. If the original sampling rate of 
%           y_obs is lower, we first interpolate y_obs to recover the missing samples.
%       (3) h_z: horizon hyper-parameter of Subspace System Identification algorithm
%           corresponding to spiking observations. Refer to Ahmadipour et al 2024 for more details.
%       (4) h_y: horizon hyper-parameter of Subspace System Identification algorithm
%           corresponding to field potential observations.
% Outputs:
%       (1) Sigma_all_N_y: Cov([N^{f}_{t};N^{p}_{t};y^{f}_{t};y^{p}_{t}]) 
%           with a size of 2*(h_z*n_z+h_y*n_y)by 2*(h_z*n_z+h_y*n_y).
%           Sigma_all_N_y can also be written in terms of [Sigma^{NN}, Sigma^{Ny}; Sigma^{Ny}', Sigma^{yy}]
%           based on definitions of Sigma^{XX} in equation set 17.
%       (2) mu_all_N_y: E([N^{f}_{t};N^{p}_{t};y^{f}_{t};y^{p}_{t})] 
%           with a size of 2*(h_z*n_z+h_y*n_y)by 1.
%           mu_all_N_y can also be written in terms of [mu^{N}; mu^{y}] based on
%           definitions of mu^{x} in equation set 17.

function [Sigma_all_N_y, mu_all_N_y] = multiscaleSIDmomentComputation(N_obs, y_obs, h_z, h_y)

    % *************************************************************************
    %               Algorithm 2 - STEP 1 - Ahmadipour et al 2024
    % *************************************************************************

    % If h_z~=h_y, the first |h_y-h_z| samples of spiking time-series (N_obs) or the field potential time-series (y_obs) are deleted as follows. 
    % This operation ensures the time indices of the first element of the future field potential activity vectors y^{f}_{t} matches those of 
    % the future spiking activity vectors N^{f}_{t} when forming the Hankel matrix (see below). 
    % This requirement can be seen from equation 14. See also Appendix A.
    if h_y - h_z >= 0
        N_obs = N_obs(:, h_y - h_z + 1:end);
        y_obs = [y_obs, nan(size(y_obs, 1), h_y - h_z)]; % To make the number of columns in the Hankel matrix (defined below) equal to T-2*h_z+1.
    else
        y_obs = y_obs(:, h_z - h_y + 1:end);
    end

    n_z = size(N_obs, 1);
    n_y = size(y_obs, 1);

    %% Computing mean of N^{f}, N^{p}, y^{f}, y^{p}
    mu_N = repmat(nanmean(N_obs, 2), 2 * h_z, 1); % mu^{N}=E(N^{f};N^{p}), see equation 17.
    mu_y = repmat(nanmean(y_obs, 2), 2 * h_y, 1); % mu^{y}=E(y^{f};y^{p}), see equation 17.
    mu_all_N_y = [mu_N; mu_y]; % concatenation of mu^{N} and mu^{y}

    T = size(N_obs, 2);

    %% Forming Hankel matrix of time-lagged spike and field observations and computing covariance of the time-lagged observations from the Hankel matrix.

    n_Hankel_cols = T - 2 * h_z + 1; % number of columns for the Hankel matrix (formed below).

    % Making the data zero mean to compute covariances.
    N_obs = N_obs - nanmean(N_obs, 2);
    y_obs = y_obs - nanmean(y_obs, 2);

    N_f = nan(h_z * n_z, n_Hankel_cols);
    N_p = nan(h_z * n_z, n_Hankel_cols);
    y_f = nan(h_y * n_y, n_Hankel_cols);
    y_p = nan(h_y * n_y, n_Hankel_cols);

    for i = 1:h_z
        N_f((i - 1) * n_z + 1:i * n_z, :) = N_obs(:, h_z + i:T - h_z + i); % Concatenating N^{f}_{t} in T-2*h_z time steps to form N^{f}.
        N_p((i - 1) * n_z + 1:i * n_z, :) = N_obs(:, h_z + 1 - i:T - h_z - i + 1); % Concatenating N^{p}_{t} in T-2*h_z time steps to form N^{p}.
    end

    for i = 1:h_y
        y_f((i - 1) * n_y + 1:i * n_y, :) = y_obs(:, h_y + i:T - 2 * h_z + h_y + i); % Concatenating y^{f}_{t} in T-2*h_z time steps to form y^{f}.
        y_p((i - 1) * n_y + 1:i * n_y, :) = y_obs(:, h_y - i + 1:T - 2 * h_z + h_y - i + 1); % Concatenating y^{p}_{t} in T-2*h_z time steps to form y^{p}.
    end

    HankelMatrix = [N_f; N_p; y_f; y_p]; % Note that HankelMatrix= [N^{fp};y^{fp}]. In Appendix A, it is shown how y^{fp}=[y^{f};y^{p}] is formed from observations.

    n_notNaN = sum(~isnan(HankelMatrix), 2); % Number of elements that are real valued (not Nan) in each row of the Hankel matrix.
    HankelMatrix(isnan(HankelMatrix)) = 0;
    temp = HankelMatrix * HankelMatrix';

    % Sigma_all_N_y = Cov( [N^{f}_{t};N^{p}_{t};y^{f}_{t};y^{p}_{t}])
    Sigma_all_N_y = nan(size(temp));

    for i = 1:size(Sigma_all_N_y, 1)

        for j = 1:size(Sigma_all_N_y, 1)
            Sigma_all_N_y(i, j) = temp(i, j) / min(n_notNaN(i), n_notNaN(j)); % temp(i,j) should be divided by the minimum number of real valued elements in row i and row j of the Hankel matrix to get a covariance value.
        end

    end

    % Making sure the computed covariance matrix is PSD and symmetric, regardless of system precision.
    [V, D] = eig(Sigma_all_N_y); D = diag(D);

    if min((D)) < 0
        D(D < 0) = 0;
        Sigma_all_N_y = V * diag(D) * V';
    end

    Sigma_all_N_y = (Sigma_all_N_y + Sigma_all_N_y') / 2;

end
