% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the prediction power (PP) as an accuracy measure
% for prediction of spiking activity for each neuron. See section 2.3.4 in
% Ahmadipour et al 2024 for more details.
% Inputs:
%       (1) N_obs: time-series of the discrete spiking activity with a size of n_z by T. n_z is
%           the number of spiking (log firing rate) signals and T is the
%           total number of time steps.
%       (2) Predicted_FR: one-step-ahead prediction of the log firing rates
%       with size n_z by T.
% Outputs:
%       (1) PP: Prediction Power of spiking activity with size n_z by T.

function PP = ComputePredictionPower(N_obs, predicted_FR)
    n_z = size(N_obs, 1); % Number of spiking signals.
    AUC = nan(1, n_z);

    for i = 1:n_z % Loop over all the available spiking signals
        spike_labels = N_obs(i, :);
        spike_labels(spike_labels > 1) = 1; % Labels are set to 0 if there is no spike in that time bin, or 1 if there is at least one spike in that time bin.
        %Computing the predicted probability of having at least one spike in each time step
        % based on the Poisson distribution for spiking activity in equation 1.
        prob_spike = 1 - exp(-predicted_FR(i, :)');
        [~, ~, ~, AUC(i)] = perfcurve(spike_labels', prob_spike, 1); % Computing Area Under the Curve of ROC (AUC)
    end

    PP = 2 * AUC' - 1; % Computing Prediction Power (PP) from AUC.
end
