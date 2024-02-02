% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function plots predictions of field activity against true field
% activity and shows the correlation coefficient between the two. 
% Inputs:
%       (1) y_true: true field data, dimension by time
%       (2) y_pred: predicted field data. This argument can also be a cell 
%               array of multiple alternative predictions of the field data 
%               using different models, in which case all alternative
%               predictions are plotted for comparison.
%       (3) Delta: the time step of sampling for the data (bin size). Used
%               to infer the right time for the plots.
%       (4) labels_pred: cell array of labels for each alternative
%               prediction in y_pred.
%       (5) indices_to_plot: index of data dimensions to plot. Default:
%               [1,2,3].
% Outputs:
%       (1) figH: handle of the generated figure

function figH = plotFieldPredictions(y_true, y_pred, Delta, labels_pred, indices_to_plot)

if ~iscell(y_pred), y_pred = {y_pred}; end

if nargin < 4, labels_pred = {}; end
if nargin < 5, indices_to_plot = []; end

%% Computing correlation coefficient between true and one-step-ahead predicted field potentials.
consecutiveNansLengths = findAllConsecutiveNansLengths(y_true(1, :));
M = consecutiveNansLengths(1) + 1; % y is available every M time steps.
n_y = size(y_true, 1); % Number of data dimensions
T = size(y_true, 2); % number of available time steps (length of data)
steps_y_available = (1:M:T); % Field potentials are available every M time steps, and the missing observations are marked with NaN values.
% y_true = data.y - repmat(params_multiscaleSID.d_y, 1, T);

CC_y = cell(length(y_pred),1);
for ind = 1:length(y_pred)
    CC_y{ind} = zeros(n_y, 1);

    for i = 1:n_y
        CC_y{ind}(i) = corr(y_pred{ind}(i, steps_y_available)', y_true(i, steps_y_available)');
    end
end

%% Plotting one-step-ahead predicted fields potentials and the original field potentials
figH = figure('Units', 'inches', 'InnerPosition', [1, 1, 8, 4]);
if isempty(indices_to_plot)
    indices_to_plot = [1, 2, 3]; % Indices of the spiking signal that we want to plot
end
subplot_counter = 0;

for i = 1:length(indices_to_plot)
    ax = subplot(length(indices_to_plot), 1, i);
    subplot_counter = subplot_counter + 1;

    if i == 1
        titleStr = sprintf('Average Correlation Coefficient (CC) between true and one-step ahead predicted fields: \n');
        for ind = 1:length(y_pred)
            if length(labels_pred) >= ind
                label = labels_pred{ind};
            else
                label = sprintf('Method %d', ind);
                labels_pred{ind} = label;
            end
            label = sprintf('%s => CC = %.4g', label, mean(CC_y{ind}));
            titleStr = sprintf('%s | %s |', titleStr, label);
        end
        title(titleStr);
        cols = get(ax,'colororder');
    end

    if i == length(indices_to_plot)
        xlabel('Time (s)');
    end

    hold on
    plot(steps_y_available * Delta, y_true(indices_to_plot(i), steps_y_available)', 'color', cols(1,:), 'LineWidth', 1.2, 'DisplayName', 'True field potentials');
    lineStyles = {'-', '--', ':'};
    for ind = 1:length(y_pred)
        plot(steps_y_available * Delta, y_pred{ind}(indices_to_plot(i), steps_y_available)', 'color', cols(1+ind, :), 'LineStyle', lineStyles{ind}, 'LineWidth', 1.2, 'DisplayName', labels_pred{ind});
    end
    legend;

    ylabel(sprintf('Example\nfield\npotential\nsignal'));

end

end