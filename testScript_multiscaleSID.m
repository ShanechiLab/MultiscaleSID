% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script runs the multiscale SID algorithm in Ahmadipour et
% al 2024 for an example simulated multimodal discrete-continuous spiking
% and field potential population activity. After the multiscale model
% parameters are learnt using multiscale SID, they are passed to the multiscale filter (MSF)
% developed in Hsieh et al 2019 to obtain one-step-ahead predictions of
% latent states and neural activity.

% Adding dependencies to the path
% Assuming CVX (http://cvxr.com/cvx/download/) has been downloaded to the 
% current directory, for example to "./CVX"
addpath(genpath('./'));

%% Setting up CVX toolbox
% make sure to execute the CVX setup script (cvx_setup.m). This script is
% is included in the CVX distribution that you should have in the path.
% cvx_setup;
cvx_startup;

%%
clear all
%% Loading multiscale simulated data
load('./simulated_data/multiscale_data', 'data_train', 'data_test', 'true_params');
%% 
n_x = size(true_params.A, 1); % latent state dimension
T_train = size(data_train.N, 2); % Training size

%% Analyzing the data to find the time scale difference of field potential and spiking observations
consecutiveNansLengths = findAllConsecutiveNansLengths(data_train.y(1, :));
assert(all(consecutiveNansLengths == consecutiveNansLengths(1)), 'Time scale difference of spikes and fields is expected to be identical across the time series. Modify the dataset!');
M = consecutiveNansLengths(1) + 1;
fprintf('Field signals are available every %d time steps.\n', M);

%% Identification of the multiscale model using multiscale SID developed in Ahmadipour et al 2024
settings = struct( ...
    'n_x', n_x, ... % A model hyperparmeter that determines the latent state dimension
    'h_z', 10, ...  % A learning setting specifying the horizon for the z-signal in Subspace System Identification (SID)
    'h_y', 10  ...  % A learning setting specifying the horizon for the z-signal in Subspace System Identification (SID)
);
tic_multiscaleSID = tic;
[params_mutiscaleSID, ~] = multiscaleSID(data_train, settings);
train_time_multiscaleSID = toc(tic_multiscaleSID);

%% Reporting training time of multiscale SID
fprintf('Training took %.3g seconds\n', train_time_multiscaleSID);

%% Plotting Identified vs True modes of the state transition matrix A
params_identified = params_mutiscaleSID; % CORRECT
figure;
eig_est = eig(params_identified.A);
eig_true = eig(true_params.A);
h(1) = scatter(real(eig_true), imag(eig_true), 50, [0, 0, 0], 'filled'); hold on;
h(2) = scatter(real(eig_est), imag(eig_est), 50, [0, 0.5, 0], 'filled'); hold on;
legends{1} = 'True modes';
legends{2} = 'Identified modes with SID';
legend(h, legends)
xlim([0.9,1]);ylim([-0.04,0.04]);
xlabel('Real'); ylabel('Imaginary');
title('True vs identified eigenvalues of A')

%% Doing Inference using multiscale filter (MSF) developed by Hsieh et al 2019.
[x_pred_test, ~, FR_pred_test, y_pred_test] = multiscaleInference(params_mutiscaleSID, data_test); % Computing and plotting of one-step-ahead prediction latent states and neural activity

% For comparison, in this simulation we can also use the true model
% parameters, which are known, to do the same inference
[x_pred_test_truemodel, ~, FR_pred_test_truemodel, y_pred_test_truemodel] = multiscaleInference(true_params, data_test); % Computing and plotting of one-step-ahead prediction latent states and neural activity

% We plot the predictions of the model learned using multiscale SID versus
% the predictions of the true model as a comparison
plotFieldPredictions(data_test.y, {y_pred_test_truemodel, y_pred_test}, data_test.Delta, {'Predicted using true model (ideal)', 'Predicted using multiscale SID'}, [1,2,3]);
plotSpikePredictions(data_test.N, {FR_pred_test_truemodel, FR_pred_test}, data_test.Delta, {'Predicted using true model (ideal)', 'Predicted using multiscale SID'}, [1,2,3]);
