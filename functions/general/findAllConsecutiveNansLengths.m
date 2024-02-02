% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2024 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Omid Sani and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function finds the lengths of all consecutive NaNs in a vector.
% Input:
%       (1) signal: a vector
% Output:
%       (1) coconsecutiveNansLengths: a vector containing the lengths of all
%       consecutive NaNs in the signal vector.
function consecutiveNansLengths = findAllConsecutiveNansLengths(signal)
    % Initialize variables
    consecutiveNansLengths = [];
    currentConsecutiveNans = 0;
    % Loop through the signal elements
    for i = 1:length(signal)

        if isnan(signal(i))
            % If NaN is encountered, increment the consecutive count
            currentConsecutiveNans = currentConsecutiveNans + 1;
        else
            % If a non-NaN value is encountered, store the consecutive count
            if currentConsecutiveNans > 0
                consecutiveNansLengths = [consecutiveNansLengths, currentConsecutiveNans];
                % Reset consecutive count
                currentConsecutiveNans = 0;
            end

        end

    end

    % Check for consecutive NaNs at the end of the signal
    if currentConsecutiveNans > 0
        consecutiveNansLengths = [consecutiveNansLengths, currentConsecutiveNans];
    end

end
