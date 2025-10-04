function [dOCV_SOC] = dOCV_SOC()
% Generate the dOCV-SOC look-up table based on OCV-SOC relation at 25°C.

% Load OCV-SOC Relation
load('OCV_SOC_relation.mat', 'OCV_SOC_25C');
% OCV_SOC_25C: [SOC (%), OCV (V)] relation at 25°C

SOC = OCV_SOC_25C(:, 1); % SOC (%)
OCV = OCV_SOC_25C(:, 2); % OCV (V)

% Interpolation for Higher Resolution
SOC_interp = linspace(0, 100, 10000);                           % High-resolution SOC range
OCV_interp = interp1(SOC, OCV, SOC_interp, 'spline'); % Interpolated OCV values

% Numerical Differentiation for dOCV/dSOC
SOC_lower_bound = 0;        % Lower SOC boundary
SOC_upper_bound = 100;   % Upper SOC boundary
num_points = 10000;           % Number of interpolation points

dOCV = gradient(OCV_interp, (SOC_upper_bound - SOC_lower_bound) / (num_points - 1)); % Gradient calculation

% Construct dOCV-SOC Look-Up Table
dOCV_SOC = [SOC_interp', dOCV'];

end


