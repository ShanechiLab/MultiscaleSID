% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function interpolates the signal with missing values marked with
% NaNs.The NaN values are replaced with interpolated values.
% Input:
%       signals: time-series with size n_signal by T, represented as
%       [signal_{1},NaN,..NaN,signal_{M+1},NaN,...NaN,signal_{2M+1},...,signal_{T}].
%       Here n_signal is the number of signals and T is the number of time steps. 
%       Signal samples are observed every M time steps and the missing samples are marked with NaNs.
% Output:
%       signals_interpolated: interpolated signals with size n_signal by (T+M-1)

function signals_interpolated = interpolateSignal(signals)
    % Create a logical mask for NaN values
    nanMask = isnan(signals(1, :));
    consecutiveNansLengths = findAllConsecutiveNansLengths(signals(1, :)); % Computing the number of consecutive NaNs in the 1st signals.
    M = consecutiveNansLengths(1) + 1; % Original signals is available every M time steps.
    % Use the mask to remove NaN values
    signals_withOutNaNs = signals(:, ~nanMask); % Removing the NaN values from the signals.
    signals_interpolated = nan(size(signals, 1), size(signals_withOutNaNs, 2) * M);

    for i = 1:size(signals, 1)
        % increasing the sample rate of each signal by a factor of M. n is half
        % the number of original sample values used to interpolate the signals.
        n = 4; cutoff = 1;
        [signals_interpolated(i, :), ~] = interp(signals_withOutNaNs(i, :), M, n, cutoff);
    end

end
