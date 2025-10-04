function [SOC_FUDS, SOC_HDS, SOC_BJDST] = SOC_measured(D_FUDS, D_HDS, D_BJDST)
% Calculate SOC from Coulomb counting for multiple driving schedules.
% The initial SOC for all tests is set to 80%.

SOC_0 = 80; % Initial SOC (%)
Q = 2;      % Battery capacity (Ah)

% Calculate SOC for each dataset
SOC_FUDS = calculate_SOC(D_FUDS, SOC_0, Q);
SOC_HDS = calculate_SOC(D_HDS, SOC_0, Q);
SOC_BJDST = calculate_SOC(D_BJDST, SOC_0, Q);

end

function SOC_result = calculate_SOC(data, SOC_0, Q)
% calculate SOC from Coulomb counting.

% Extract time, current, from data
t = data(:, 1);
I = data(:, 2);
n = length(I);

% Initialize SOC array
SOC = zeros(n, 1);
SOC(1) = SOC_0;

% Coulomb counting
for i = 1:n-1
    dt = t(i+1) - t(i); % Time step
    SOC(i+1) = SOC(i) + (dt * I(i) / (Q * 3600)) * 100; % Update SOC
end

% time and SOC for output
SOC_result = [t, SOC];

end

