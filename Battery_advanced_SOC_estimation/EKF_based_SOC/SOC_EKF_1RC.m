% Extended Kalman Filter for State of Charge (SOC) Estimation
clear; clc;

% Load and Read Data
% Load dynamic driving data
[D_FUDS, D_HDS, D_BJDST] = Read_dynamic_data();
%   D_FUDS: Federal Urban Driving Schedule data [Time(s), Current(A), Voltage(V)]
%   D_HDS: Highway Driving Schedule data
%   D_BJDST: Beijing Dynamic Stress Test data
time_1 = D_HDS(:, 1);  
Current_1=  D_HDS(:, 2);
Voltage_1 = D_HDS(:, 3);

figure(5)
plot(time_1,Current_1)
xlabel("Time(s)")
ylabel("Current")
title("Current profile")


figure(6)
plot(time_1,Voltage_1)
xlabel("Time(s)")
ylabel("Voltage")
title("Terminal voltage")


time_2 = D_HDS(:, 1);  
Current_2=  D_HDS(:, 2);
Voltage_2 = D_HDS(:, 3);

% Load OCV-SOC and build the dOCV-SOC relation
load('OCV_SOC_relation.mat');   % OCV-SOC look-up table (25°C)

dOCV_SOC = dOCV_SOC();      % dOCV-SOC look-up table

% Compute SOC using Coulomb counting (experimental results)
[SOC_FUDS, SOC_HDS, SOC_BJDST] = SOC_measured(D_FUDS, D_HDS, D_BJDST);   % Effective range: 10-80%


% Model parameters (1RC model)
 x_P = [0.070248, 0.009953, 885.996888];    % PSO 1
%x_P = [0.07219, 0.01193, 94836.46270];    % PSO bounds from pulse test
 %x_P = [0.07259, 0.03197, 1413.22938];    % from Least Squares method
% x_P = [0.08191, 0.02386, 47418.23135];  % from pulse test method

% Extended Kalman Filter Implementation
% SOC estimation using EKF
SOCdata = SOC_HDS;       % SOC_FUDS, SOC_HDS, or SOC_BJDST, Measured SOC
D = D_HDS;                        % D_FUDS, D_HDS, or D_BJDST, the used data
[x_hat_plus] = SOC_model_EKF(D, OCV_SOC_25C, dOCV_SOC, x_P);

% Extract and process results
SOC_model = x_hat_plus(2, :)' * 100;    % SOC from model (as percentage)


t = SOCdata(:, 1);            % Time (s)
SOC_measure = SOCdata(:, 2);     % SOC_FUDS, SOC_HDS, or SOC_BJDST, Measured SOC
% Calculate Root Mean Square Error (RMSE)
n = length(SOC_model);    
RMSE = sqrt(mean((SOC_measure(1:n) - SOC_model(1:n)).^2));
% MAE = mean((SOC_measure(1:n) - SOC_model(1:n)));
% absolute_error =     abs(y_true - y_pred);
MAE = mean( abs(SOC_measure(1:n) - SOC_model(1:n)));

% Plot Results
figure;
plot(t, SOC_measure, 'LineWidth', 2, 'DisplayName', 'Measured SOC');
hold on;
plot(t(1:n), SOC_model(1:n), '--', 'LineWidth', 2, 'DisplayName', 'Estimated SOC');
hold off;
grid on; 
 % xlim([0 2000])
xlabel('Time (s)', 'FontSize', 18);
ylabel('SOC (%)', 'FontSize', 18);
title(['SOC Estimation with EKF(1RC) (MAE = ' num2str(MAE, '%.2f') '%)']);
legend('show', 'FontSize', 15);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', 15);


%% SOC_EKF function

function [x_hat_plus] = SOC_model_EKF(Data, OCV_SOC_25C, dOCV_SOC, x_P)
% Extended Kalman Filter to predict SOC of a Lithium-ion cell.
% Outputs SOC and transient voltage (V1) from the state estimates.

% Assign Data
t = Data(:, 1);   % Time (s)
I = Data(:, 2);  % Current (A)
V = Data(:, 3); % Measured Voltage (V)
n = length(I);
SOC_initial = 0.7; % Initial SOC guess


%%% simulate measured current signal by adding noises
noise_std = 0.0;            % Standard deviation of the noise
noise_mean = 0.0;        % Mean of the noise
noise = noise_std * randn(size(I)) + noise_mean;   % Generate noise with a non-zero mean
I = I + noise;                  % Add noise to the signal 
%%%


% Model Parameters
R0 = x_P(1); R1 = x_P(2); C1 = x_P(3);
capacity = 2;  % Battery capacity (Ah)

% Kalman Initialization
V1_0 = 0;                      % Initial transient voltage guess (V1)
SOC_0 = SOC_initial;   % Initial SOC guess
x_hat_0 = [V1_0; SOC_0]; % Initial state

P0 = diag([5e-5, 5e-2]);  % Initial estimation error covariance
Q = diag([1e-6, 1e-5]);    % Process noise covariance
R = 6e-2;                         % Measurement noise variance

x_hat_plus = zeros(2, n);   % Preallocate state estimates
K = zeros(2, n);                  % Preallocate Kalman gain
err = zeros(1, n);                % Preallocate error vector

% EKF Loop
for i = 1:n

% time interval    
   if i == 1; dt = t(i);
   else; dt = t(i)-t(i-1); end

    % State transition and input matrices
    A = [exp(-dt / (R1 * C1)), 0; 0, 1];
    B = [R1 * (1 - exp(-dt / (R1 * C1))); dt / (capacity * 3600)];

% Prediction step
   if i == 1 
      x_hat_minus = A*x_hat_0 + B*I(i);     % State estimate.  
      P_minus = A*P0*A' + Q;                     % Estimation-error covariance.
   else
      x_hat_minus = A*x_hat_plus(:,i-1) + B*I(i);   % State estimate.  
      P_minus = A*P_plus*A' + Q;                         % Estimation-error covariance.  
   end


% Update step
    % Interpolate OCV and dOCV
    SOC = x_hat_minus(2);
    OCV = interp1(OCV_SOC_25C(:, 1) / 100, OCV_SOC_25C(:, 2), SOC, 'spline');
    dOCV = interp1(dOCV_SOC(:, 1) / 100, dOCV_SOC(:, 2), SOC, 'spline');

     C = [1, dOCV];
%   C = [1, 0];

    % Kalman gain
    K(:, i) = P_minus * C' / (C * P_minus * C' + R);

   % Measurement model    
    y_est = OCV + R0 * I(i) + x_hat_minus(1);

    % Measurement update
    err(i) = V(i) - y_est;      % Voltage prediction error
    x_hat_plus(:, i) = x_hat_minus + K(:, i) * err(i);   % update the state estimate
    P_plus = (eye(2) - K(:, i) * C) * P_minus;            % update the error covariance
    Pv1(i) = P_plus(1,1);
    Psoc(i) = P_plus(2,2);
end

end