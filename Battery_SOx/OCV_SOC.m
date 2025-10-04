
function [OCV_SOC_25C] = OCV_SOC ()
% To determine the relationship between OCV and SOC
% From low current test

% Load data
[num, ~] = xlsread('OCV_SOC_25C_lowCurrentSample1.xlsx');
t = num(:, 1);           % Test time (s)
I = num(:, 4) / 1000;    % Current (A)
V = num(:, 3) / 1000;    % Voltage (V)

% Extract OCV data for charge and discharge processes
OCV_Dis = V(7348:85324);
OCV_Cc = V(92524:162212);

% Compute discharge capacity (Dc)
Dc = cumsum(((I(7348:85323) + I(7349:85324)) / 2) .* diff(t(7348:85324)) / 3600);

% Compute charge capacity (Cc)
Cc = cumsum(((I(92524:162211) + I(92525:162212)) / 2) .* diff(t(92524:162212)) / 3600);

% Calculate SOC for charge and discharge
SOC_Dis = 1 - Dc / Dc(end);
SOC_Cc = Cc / Cc(end);

% Combine SOC and OCV for charge and discharge
OCV_SOC_Cha_25C = [SOC_Cc * 100, OCV_Cc(1:end-1)];
OCV_SOC_Dis_25C = [SOC_Dis * 100, OCV_Dis(1:end-1)];

% Process charge data and ensure unique SOC values
[~, unique_idx] = unique(OCV_SOC_Cha_25C(:,1), 'stable');    % 'stable' keeps the order of the first occurrence
OCV_SOC_Cha_25C = OCV_SOC_Cha_25C(unique_idx, :);

[~, unique_idx] = unique(OCV_SOC_Cha_25C(:,2), 'stable');    % 'stable' keeps the order of the first occurrence
OCV_SOC_Cha_25C = OCV_SOC_Cha_25C(unique_idx, :);

x_cha = OCV_SOC_Cha_25C(:, 1);
v_cha = OCV_SOC_Cha_25C(:, 2);

% Process discharge data and ensure unique SOC values
[~, unique_idx] = unique(OCV_SOC_Dis_25C(:,1), 'stable');     % 'stable' keeps the order of the first occurrence
OCV_SOC_Dis_25C = OCV_SOC_Dis_25C(unique_idx, :);

[~, unique_idx] = unique(OCV_SOC_Dis_25C(:,2), 'stable');     % 'stable' keeps the order of the first occurrence
OCV_SOC_Dis_25C = OCV_SOC_Dis_25C(unique_idx, :);

x_dis = OCV_SOC_Dis_25C(:, 1);
v_dis = OCV_SOC_Dis_25C(:, 2);

% Interpolation for charge and discharge
xq = 0:0.05:100;
vq_cha = interp1(x_cha, v_cha, xq, 'linear', 'extrap');
vq_dis = interp1(x_dis, v_dis, xq, 'linear', 'extrap');

% Calculate average OCV
OCV_SOC_25C = [xq', (vq_cha' + vq_dis') / 2];


% Plot results
figure;

% Set background color to white
set(gcf, 'Color', 'w'); 

% Plot the Charge curve
plot(OCV_SOC_Cha_25C(:, 1), OCV_SOC_Cha_25C(:, 2), 'LineWidth', 2, 'DisplayName', 'Charge');
hold on;

% Plot the Discharge curve
plot(OCV_SOC_Dis_25C(:, 1), OCV_SOC_Dis_25C(:, 2), 'LineWidth', 2, 'DisplayName', 'Discharge');

% Plot the Average curve
plot(OCV_SOC_25C(:, 1), OCV_SOC_25C(:, 2), 'k', 'LineWidth', 2, 'DisplayName', 'Average');
hold off;

grid on;  % Enable grid
set(gca, 'FontSize', 15); % Set axis font size
ylim ([2.5 4.2])
% Add title and axis labels
title('OCV-SOC-25C-Low Current', 'FontSize', 20); % Font size increased to 16
xlabel('SOC (%)', 'FontSize', 18); 
ylabel('OCV (V)', 'FontSize', 18); 
legend('FontSize', 16); 

end


