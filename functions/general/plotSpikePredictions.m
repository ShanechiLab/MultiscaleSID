% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function plots predictions of firing rate of spiking activity 
% against true spiking activity and shows the prediction power of the
% firing rates.
% Inputs:
%       (1) N_true: true spiking data, dimension by time
%       (2) FR_pred: predicted spiking rates. This argument can also be a cell 
%               array of multiple alternative predictions of the spike rates
%               using different models, in which case all alternative
%               predictions are plotted for comparison.
%       (3) Delta: the time step of sampling for the data (bin size). Used
%               to infer the right time for the plots.
%       (4) labels_pred: cell array of labels for each alternative
%               prediction in FR_pred.
%       (5) indices_to_plot: index of data dimensions to plot. Default:
%               [1,2,3].
% Outputs:
%       (1) figH: handle of the generated figure

function figH = plotSpikePredictions(N_true, FR_pred, Delta, labels_pred, indices_to_plot)

if ~iscell(FR_pred), FR_pred = {FR_pred}; end

if nargin < 4, labels_pred = {}; end
if nargin < 5, indices_to_plot = []; end

% Computing the predicted probability of having at least one spike in each time step
% based on the Poisson distribution for spiking activity
spike_prob = cell(size(FR_pred));
PP_N = cell(size(FR_pred));
for ind = 1:length(spike_prob)
    PP_N{ind} = ComputePredictionPower(N_true, FR_pred{ind});
    spike_prob{ind} = 1 - exp(-1 * FR_pred{ind});
end
T = size(N_true, 2); % number of available time steps (length of data)
time_indices = 1:T;

%% Computing Prediction Power (PP) of spiking activity (see section 2.3.4 for details).

%% Plotting predicted spiking probability and the original spiking activity
figH = figure('Units', 'inches', 'InnerPosition', [1, 1, 8, 4]);
if isempty(indices_to_plot)
    indices_to_plot = [1, 2, 3]; % Indices of the spiking signal that we want to plot
end
subplot_counter = 0;

for i = 1:length(indices_to_plot)
    ax = subplot(length(indices_to_plot), 1, i);
    subplot_counter = subplot_counter + 1;

    if i == 1
        titleStr = sprintf('Average Prediction Power (PP) for spiking activity: \n');
        for ind = 1:length(spike_prob)
            if length(labels_pred) >= ind
                label = labels_pred{ind};
            else
                label = sprintf('Method %d', ind);
                labels_pred{ind} = label;
            end
            label = sprintf('%s => PP = %.4g', label, mean(PP_N{ind}));
            titleStr = sprintf('%s | %s |', titleStr, label);
        end
        title(titleStr);
        cols = get(ax,'colororder');
    end

    if i == length(indices_to_plot)
        xlabel('Time (s)');
    end

    hold on
    steps_spike = find(N_true(indices_to_plot(i), time_indices) >= 1);

    for j = 1:length(steps_spike)
        h1 = plot([steps_spike(j), steps_spike(j)] * Delta, [0, 1], 'color', cols(1,:), 'LineWidth', 0.2, 'DisplayName', 'True spikes');
    end

    lineStyles = {'-', '--', ':'};
    legHs = [h1];
    for ind = 1:length(spike_prob)
        h2 = plot([1:length(time_indices)] * Delta, spike_prob{ind}(indices_to_plot(i), time_indices), 'color', cols(1+ind, :), 'LineStyle', lineStyles{ind}, 'LineWidth', 1.2, 'DisplayName', labels_pred{ind});
        legHs = [legHs, h2];
    end
    legend(legHs);

    yticks([0:0.2:1]);
    ylim([0, max(spike_prob{end}(indices_to_plot(i), time_indices))]);

    ylabel(sprintf('Example\npredicted\nspiking\nprobability'));

end

end