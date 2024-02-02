% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function obtains the one-step-ahead prediction of spike and field
% potential activity. To obtain them, we need to first obtain the
% one-step-ahead prediction of latent states. To do so, we use the
% identified model parameter by our multiscale SID to construct the
% multiscale filter (MSF) to obtain the one-step-ahead prediction of latent
% states. The MSF is derived in "Multiscale modeling and decoding
% algorithms for spike-field activity" by Hsieh et al 2019. The MSF is
% implemented in the "Decoder.m" Matlab function by Abbaspourazad et al,
% which is used below 
% (available also here https://github.com/ShanechiLab/multiscaleEM).
% Inputs:
%       (1) params_multiscaleSID: a structure whose fields represent parameters
%               identified by multiscale SID, i.e. the set N={A, C_z, C_y, Q,
%               R_z,d_z, d_y}
%       (2) data: a structure containing the neural observation with the following fields:
%           - N: discrete spiking time series [N_1, N_2,...,N_T] with a 
%               size of n_z by T. n_z is the number of spiking (log firing rate) 
%               signals and T is the total number of time steps.
%           - y: continuous Gaussian times-series (e.g. field potential activity) 
%               represented as [y1,NaN,...,NaN, y_{M+1}, NaN,...,NaN,y_{2M+1},...,y_{T}] 
%               with size of n_y by T. 
%               Here, n_y is the number of field signals. Field signals are 
%               observed every M time steps and the missing observations 
%               are marked with NaNs.
%       (3) generate_plots: generates plots of predictions.
% Outputs:
%       (1) x_pred: A matrix containing x_{t|t-1} for all t in T, with size n_x by T.
%               x_{t|t-1} are filtered values of latent states at time t,
%               using observations up to time t-1.
%       (2) x_upd: A matrix containing x_{t|t} for all t in T, with size n_x by T. 
%               x_{t|t} are filtered values of latent states at time t,
%               using observations up to time t.
%       (3) FR_pred: A matrix containing z_{t|t-1}=C_z*x_{t|t-1}+d_z for 
%               all t in T, with size n_z by T
%       (4) y_pred: A matrix containing y_{t|t-1}=C_y*x_{t|t-1} for 
%               all t in T, with size n_y by T.

function [x_pred, x_upd, FR_pred, y_pred] = multiscaleInference(params_multiscaleSID, data, generate_plots)

    if nargin < 3, generate_plots = false; end

    consecutiveNansLengths = findAllConsecutiveNansLengths(data.y(1, :));
    M = consecutiveNansLengths(1) + 1; % y is available every M time steps.
    n_x = size(params_multiscaleSID.A, 1); % dimension of the latent state
    n_y = size(params_multiscaleSID.C_y, 1); % dimension of the y observations (number of y signals)
    Delta = data.Delta; % timescale of dynamics, or sampling in seconds

    %% Converting the model parameters learnt by multiscale SID to the model parameters of MSF.
    rem_samples = 1; % The MSF implementation in the decoder.m Matlab function, assumes y observations are available at M,2M,3M,..., while the provided data has observations at 1,M+1,2M+1. So we will remove the first sample.
    y_tmp = data.y(:, (1+rem_samples):end); % 
    N_tmp = data.N(:, (1+rem_samples):end); %
    T_tmp = size(N_tmp, 2); % number of available time steps (length of data)
    params_MSF = convertParams_multiscaleSIDtoMSF(params_multiscaleSID, data.Delta);

    %% Obtaining one-step-ahead prediction and estimation of latent states using MSF, from neural observations and the learnt multiscale model parameters
    settings.Scale_dif = M; settings.delta = data.Delta; settings.Input = zeros(1, T_tmp);
    [x_upd, x_pred, ~, ~] = Decoder(params_MSF.A, zeros(n_x, 1), params_MSF.Q, zeros(n_x, 1), zeros(n_x, n_x), params_MSF.C, zeros(n_y, 1), params_MSF.R, params_MSF.Theta, y_tmp - repmat(params_MSF.Bias, 1, T_tmp), N_tmp, settings);

    %% Add back any removed samples to predictions to have consistent input and output dimensions
    if rem_samples > 0
        x_upd = [zeros(size(x_upd,1), rem_samples), x_upd];
        x_pred = [zeros(size(x_pred,1), rem_samples), x_pred];
    end

    %% Computing Prediction Power (PP) of spiking activity (see section 2.3.4 for details).
    FR_pred = exp(params_multiscaleSID.C_z * x_pred + params_multiscaleSID.d_z); % Computing one-step-ahead prediction of the log firing rate based on z_{t|t-1}=C_z*x_{t|t-1}+d_z
    PP_N = ComputePredictionPower(data.N, FR_pred);

    %% Computing correlation coefficient between true and one-step-ahead predicted field potentials.
    T = size(data.N, 2); % number of available time steps (length of data)
    steps_y_available = (1:M:T); % Field potentials are available every M time steps, and the missing observations are marked with NaN values.
    y_pred = params_multiscaleSID.C_y * x_pred; % Computing one-step-ahead prediction of firing rate based on y_{t|t-1}=C_y*x_{t|t-1}
    y_pred = y_pred + repmat(params_multiscaleSID.d_y, 1, T);

    if generate_plots
        y_true = data.y;

        CC_y = zeros(n_y, 1);
    
        for i = 1:n_y
            CC_y(i) = corr(y_pred(i, steps_y_available)', y_true(i, steps_y_available)');
        end

        %% Plotting one-step-ahead predicted fields potentials and the original field potentials
        figure
        indices_plot_y = [1, 2, 3]; % Indices of the spiking signal that we want to plot
        subplot_counter = 0;
    
        for i = 1:length(indices_plot_y)
            subplot(length(indices_plot_y), 1, i)
            subplot_counter = subplot_counter + 1;
    
            if i == 1
                title(sprintf('Average Correlation Coefficient (CC) between true and one-step ahead predicted fields is %.4g', mean(CC_y)));
            end
    
            if i == length(indices_plot_y)
                xlabel('Time (s)');
            end
    
            hold on
            plot(steps_y_available * Delta, y_true(indices_plot_y(i), steps_y_available)', 'b', 'LineWidth', 1.2);
            plot(steps_y_available * Delta, y_pred(indices_plot_y(i), steps_y_available)', 'r', 'LineWidth', 1.2);
            legend({'Field potential', 'Predicted field potential'});
    
            ylabel(sprintf('Example\nfield\npotential\nsignal'));
            fig = gcf;
            newWidth = 800; newHeight = 300;
            set(fig, 'Position', [fig.Position(1), fig.Position(2), newWidth, newHeight]);
    
        end

        %% Plotting predicted spiking probability and the original spiking activity
        figure
        indices_plot_N = [1, 2, 3]; % Indices of the spiking signal that we want to plot
    
        % Computing the predicted probability of having at least one spike in each time step
        % based on the Poisson distribution for spiking activity
        spike_prob = 1 - exp(-1 * FR_pred);
        time_indices = 1:T;
    
        subplot_counter = 0;
    
        for i = 1:length(indices_plot_N)
            subplot(length(indices_plot_N), 1, i)
            subplot_counter = subplot_counter + 1;
    
            if i == 1
                title(sprintf('Average Prediction Power (PP) for spiking activity is %.4g', mean(PP_N)));
            end
    
            if i == length(indices_plot_N)
                xlabel('Time (s)');
            end
    
            hold on
            steps_spike = find(data.N(indices_plot_N(i), time_indices) >= 1);
    
            for j = 1:length(steps_spike)
                h1 = plot([steps_spike(j), steps_spike(j)] * Delta, [0, 1], 'b', 'LineWidth', 0.2);
            end
    
            ylabel(sprintf('Example\npredicted\nspiking\nprobability'));
            h2 = plot([1:length(time_indices)] * Delta, spike_prob(indices_plot_N(i), time_indices), 'r', 'LineWidth', 1.2);
            legend([h1,h2], {'Spikes', 'Predicted spiking probability'});
            yticks([0:0.2:1]);
            ylim([0, max(spike_prob(indices_plot_N(i), time_indices))]);
            fig = gcf;
            newWidth = 800;
            newHeight = 300;
            set(fig, 'Position', [fig.Position(1), fig.Position(2), newWidth, newHeight]);
        end
    end

end
