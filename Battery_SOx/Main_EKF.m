% Extended Kalman Filter for State of Charge (SOC) Estimation
clear; clc;

% Load and Read Data
% Load dynamic driving data
[D_FUDS, D_HDS, D_BJDST] = Read_dynamic_data();
%   D_FUDS: Federal Urban Driving Schedule data [Time(s), Current(A), Voltage(V)]
%   D_HDS: Highway Driving Schedule data
%   D_BJDST: Beijing Dynamic Stress Test data

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
SOCdata = SOC_BJDST;       % SOC_FUDS, SOC_HDS, or SOC_BJDST, Measured SOC
D = D_BJDST;                        % D_FUDS, D_HDS, or D_BJDST, the used data
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
title(['SOC Estimation with EKF (MAE = ' num2str(MAE, '%.2f') '%)']);
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

% % 
% % 
% % 
% % 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% chnage in initial SOC guess %%%%%%%%%%%%%%%%

% % % Load and Read Data
% % % Load dynamic driving data
% % [D_FUDS, D_HDS, D_BJDST] = Read_dynamic_data();
% % %   D_FUDS: Federal Urban Driving Schedule data [Time(s), Current(A), Voltage(V)]
% % %   D_HDS: Highway Driving Schedule data
% % %   D_BJDST: Beijing Dynamic Stress Test data
% % 
% % % Load OCV-SOC and build the dOCV-SOC relation
% % load('OCV_SOC_relation.mat');   % OCV-SOC look-up table (25°C)
% % 
% % dOCV_SOC = dOCV_SOC();      % dOCV-SOC look-up table
% % 
% % % Compute SOC using Coulomb counting (experimental results)
% % [SOC_FUDS, SOC_HDS, SOC_BJDST] = SOC_measured(D_FUDS, D_HDS, D_BJDST);   % Effective range: 10-80%
% % 
% % % Model parameters (1RC model)
% % x_P = [0.070248, 0.009953, 885.996888];    % PSO 1
% % 
% % % Extended Kalman Filter Implementation
% % SOCdata = SOC_FUDS;       % SOC_FUDS, SOC_HDS, or SOC_BJDST, Measured SOC
% % D = D_FUDS;                        % D_FUDS, D_HDS, or D_BJDST, the used data
% % 
% % % Define different initial SoC values
% % SOC_initial_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1];
% % n_variants = length(SOC_initial_values);
% % SOC_model_all = cell(n_variants, 1);
% % 
% % figure; hold on;
% % 
% % for j = 1:n_variants
% %     SOC_initial = SOC_initial_values(j);
% %     [x_hat_plus] = SOC_model_EKF(D, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial);
% %     SOC_model = x_hat_plus(2, :)' * 100;    % SOC from model (as percentage)
% %     SOC_model_all{j} = SOC_model;
% %     plot(SOCdata(:, 1), SOC_model, 'LineWidth', 2, 'DisplayName', ['Init SOC = ' num2str(SOC_initial*100) '%']);
% % end
% % 
% % % Plot Measured SOC for reference
% % plot(SOCdata(:, 1), SOCdata(:, 2), 'k', 'LineWidth', 2, 'DisplayName', 'Measured SOC');
% % 
% % grid on; 
% % xlabel('Time (s)', 'FontSize', 18);
% % ylabel('SOC (%)', 'FontSize', 18);
% % title('SOC Estimation with EKF for Different Initial SOC Values');
% % legend('show', 'FontSize', 15);
% % set(gcf, 'Color', 'w');
% % set(gca, 'FontSize', 15);
% % hold off;
% % 
% % %% Updated SOC_EKF function
% % function [x_hat_plus] = SOC_model_EKF(Data, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial)
% % % Extended Kalman Filter to predict SOC of a Lithium-ion cell.
% % % Outputs SOC and transient voltage (V1) from the state estimates.
% % 
% % % Assign Data
% % t = Data(:, 1);   % Time (s)
% % I = Data(:, 2);  % Current (A)
% % V = Data(:, 3); % Measured Voltage (V)
% % n = length(I);
% % 
% % %%% simulate measured current signal by adding noises
% % noise_std = 0.0;            % Standard deviation of the noise
% % noise_mean = 0.0;        % Mean of the noise
% % noise = noise_std * randn(size(I)) + noise_mean;   % Generate noise with a non-zero mean
% % I = I + noise;                  % Add noise to the signal 
% % %%%
% % 
% % % Model Parameters
% % R0 = x_P(1); R1 = x_P(2); C1 = x_P(3);
% % capacity = 2;  % Battery capacity (Ah)
% % 
% % % Kalman Initialization
% % V1_0 = 0;                      % Initial transient voltage guess (V1)
% % SOC_0 = SOC_initial;   % Initial SOC guess
% % x_hat_0 = [V1_0; SOC_0]; % Initial state
% % 
% % P0 = diag([5e-5, 5e-2]);  % Initial estimation error covariance
% % Q = diag([1e-6, 1e-5]);    % Process noise covariance
% % R = 6e-2;                         % Measurement noise variance
% % 
% % x_hat_plus = zeros(2, n);   % Preallocate state estimates
% % K = zeros(2, n);                  % Preallocate Kalman gain
% % err = zeros(1, n);                % Preallocate error vector
% % 
% % % EKF Loop
% % for i = 1:n
% % 
% % % time interval    
% %    if i == 1; dt = t(i);
% %    else; dt = t(i)-t(i-1); end
% % 
% %     % State transition and input matrices
% %     A = [exp(-dt / (R1 * C1)), 0; 0, 1];
% %     B = [R1 * (1 - exp(-dt / (R1 * C1))); dt / (capacity * 3600)];
% % 
% % % Prediction step
% %    if i == 1 
% %       x_hat_minus = A*x_hat_0 + B*I(i);     % State estimate.  
% %       P_minus = A*P0*A' + Q;                     % Estimation-error covariance.
% %    else
% %       x_hat_minus = A*x_hat_plus(:,i-1) + B*I(i);   % State estimate.  
% %       P_minus = A*P_plus*A' + Q;                         % Estimation-error covariance.  
% %    end
% % 
% % % Update step
% %     % Interpolate OCV and dOCV
% %     SOC = x_hat_minus(2);
% %     OCV = interp1(OCV_SOC_25C(:, 1) / 100, OCV_SOC_25C(:, 2), SOC, 'spline');
% %     dOCV = interp1(dOCV_SOC(:, 1) / 100, dOCV_SOC(:, 2), SOC, 'spline');
% % 
% %     C = [1, dOCV];
% % 
% %     % Kalman gain
% %     K(:, i) = P_minus * C' / (C * P_minus * C' + R);
% % 
% %    % Measurement model    
% %     y_est = OCV + R0 * I(i) + x_hat_minus(1);
% % 
% %     % Measurement update
% %     err(i) = V(i) - y_est;      % Voltage prediction error
% %     x_hat_plus(:, i) = x_hat_minus + K(:, i) * err(i);   % update the state estimate
% %     P_plus = (eye(2) - K(:, i) * C) * P_minus;            % update the error covariance
% % end
% % 
% % end
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% chnage the noise_std%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Load and Read Data
% Load dynamic driving data
% [D_FUDS, D_HDS, D_BJDST] = Read_dynamic_data();
% load('OCV_SOC_relation.mat');   % OCV-SOC look-up table (25°C)
% dOCV_SOC = dOCV_SOC();          % dOCV-SOC look-up table
% 
% % Compute SOC using Coulomb counting (experimental results)
% [SOC_FUDS, SOC_HDS, SOC_BJDST] = SOC_measured(D_FUDS, D_HDS, D_BJDST);   % Effective range: 10-80%
% 
% % Model parameters (1RC model)
% x_P = [0.070248, 0.009953, 885.996888];  % PSO 1
% 
% % Define different noise standard deviations
% noise_std_values = [1,1.5,2.0,2.5,3.0];
% 
% % Fixed Initial SoC
% SOC_initial = 0.7;  
% 
% % Select dataset
% SOCdata = SOC_FUDS;       
% D = D_FUDS;      
% 
% % Time vector
% t = SOCdata(:, 1);           
% SOC_measure = SOCdata(:, 2);    
% 
% % Initialize figure
% figure; hold on; grid on;
% 
% % Loop through different noise levels
% for idx = 1:length(noise_std_values)
%     noise_std = noise_std_values(idx);
% 
%     % Run EKF SOC estimation with different noise_std values
%     [x_hat_plus] = SOC_model_EKF(D, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std);
%     SOC_model = x_hat_plus(2, :)' * 100;  
% 
%     % Plot results
%     plot(t, SOC_model, 'LineWidth', 2, 'DisplayName', ['Noise std = ' num2str(noise_std)]);
% end
% 
% % Plot measured SOC for reference
% plot(t, SOC_measure, 'k--', 'LineWidth', 2, 'DisplayName', 'Measured SOC');
% 
% % Format plot
% xlabel('Time (s)', 'FontSize', 18);
% ylabel('SOC (%)', 'FontSize', 18);
% title('Effect of Noise on SoC Estimation (EKF)', 'FontSize', 18);
% legend('show', 'FontSize', 15);
% set(gcf, 'Color', 'w');
% set(gca, 'FontSize', 15);
% 
% %% Updated SOC_EKF function
% 
% function [x_hat_plus] = SOC_model_EKF(Data, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std)
% % Extended Kalman Filter to predict SOC of a Lithium-ion cell.
% % Inputs:
% %   - Data: Driving profile data
% %   - OCV_SOC_25C: OCV-SOC lookup table
% %   - dOCV_SOC: dOCV-SOC lookup table
% %   - x_P: Model parameters
% %   - SOC_initial: Initial SoC
% %   - noise_std: Standard deviation of noise to be added
% 
% % Assign Data
% t = Data(:, 1);  
% I = Data(:, 2);  
% V = Data(:, 3);  
% n = length(I);
% 
% % Add noise to current signal
% noise_mean = 0.0;
% noise = noise_std * randn(size(I)) + noise_mean;
% I = I + noise;
% 
% % Model Parameters
% R0 = x_P(1); R1 = x_P(2); C1 = x_P(3);
% capacity = 2;  
% 
% % Kalman Initialization
% V1_0 = 0;                     
% SOC_0 = SOC_initial;  
% x_hat_0 = [V1_0; SOC_0];  
% 
% P0 = diag([5e-5, 5e-2]);  
% Q = diag([1e-6, 1e-5]);    
% R = 6e-2;                         
% 
% x_hat_plus = zeros(2, n);   
% K = zeros(2, n);                
% 
% % EKF Loop
% for i = 1:n
%     if i == 1; dt = t(i);
%     else; dt = t(i)-t(i-1); end
% 
%     % State transition and input matrices
%     A = [exp(-dt / (R1 * C1)), 0; 0, 1];
%     B = [R1 * (1 - exp(-dt / (R1 * C1))); dt / (capacity * 3600)];
% 
%     % Prediction step
%     if i == 1 
%         x_hat_minus = A*x_hat_0 + B*I(i);    
%         P_minus = A*P0*A' + Q;                    
%     else
%         x_hat_minus = A*x_hat_plus(:,i-1) + B*I(i);  
%         P_minus = A*P_plus*A' + Q;                         
%     end
% 
%     % Interpolate OCV and dOCV
%     SOC = x_hat_minus(2);
%     OCV = interp1(OCV_SOC_25C(:, 1) / 100, OCV_SOC_25C(:, 2), SOC, 'spline');
%     dOCV = interp1(dOCV_SOC(:, 1) / 100, dOCV_SOC(:, 2), SOC, 'spline');
% 
%     C = [1, dOCV];
% 
%     % Kalman gain
%     K(:, i) = P_minus * C' / (C * P_minus * C' + R);
% 
%     % Measurement model    
%     y_est = OCV + R0 * I(i) + x_hat_minus(1);
% 
%     % Measurement update
%     err = V(i) - y_est;     
%     x_hat_plus(:, i) = x_hat_minus + K(:, i) * err;   
%     P_plus = (eye(2) - K(:, i) * C) * P_minus;            
% end
% 
% end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%% change in noise mean %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

% % Load and Read Data
% % Load dynamic driving data
% [D_FUDS, D_HDS, D_BJDST] = Read_dynamic_data();
% load('OCV_SOC_relation.mat');   % OCV-SOC look-up table (25°C)
% dOCV_SOC = dOCV_SOC();          % dOCV-SOC look-up table
% 
% % Compute SOC using Coulomb counting (experimental results)
% [SOC_FUDS, SOC_HDS, SOC_BJDST] = SOC_measured(D_FUDS, D_HDS, D_BJDST);   % Effective range: 10-80%
% 
% % Model parameters (1RC model)
% x_P = [0.070248, 0.009953, 885.996888];  % PSO 1
% 
% % Define different noise mean values
% noise_mean_values = [0.01,0.02,0.03,0.04,0.05];
% 
% % Fixed Initial SoC and noise standard deviation
% SOC_initial = 0.7;  
% noise_std = 0.02;  
% 
% % Select dataset
% SOCdata = SOC_FUDS;       
% D = D_FUDS;      
% 
% % Time vector
% t = SOCdata(:, 1);           
% SOC_measure = SOCdata(:, 2);    
% 
% % Initialize figure
% figure; hold on; grid on;
% 
% % Loop through different noise mean values
% for idx = 1:length(noise_mean_values)
%     noise_mean = noise_mean_values(idx);
% 
%     % Run EKF SOC estimation with different noise_mean values
%     [x_hat_plus] = SOC_model_EKF(D, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std, noise_mean);
%     SOC_model = x_hat_plus(2, :)' * 100;  
% 
%     % Plot results
%     plot(t, SOC_model, 'LineWidth', 2, 'DisplayName', ['Noise mean = ' num2str(noise_mean)]);
% end
% 
% % Plot measured SOC for reference
% plot(t, SOC_measure, 'k--', 'LineWidth', 2, 'DisplayName', 'Measured SOC');
% 
% % Format plot
% xlabel('Time (s)', 'FontSize', 18);
% ylabel('SOC (%)', 'FontSize', 18);
% title('Effect of Bias Noise on SoC Estimation (EKF)', 'FontSize', 18);
% legend('show', 'FontSize', 15);
% set(gcf, 'Color', 'w');
% set(gca, 'FontSize', 15);
% 
% %% Updated SOC_EKF function
% 
% function [x_hat_plus] = SOC_model_EKF(Data, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std, noise_mean)
% % Extended Kalman Filter to predict SOC of a Lithium-ion cell.
% % Inputs:
% %   - Data: Driving profile data
% %   - OCV_SOC_25C: OCV-SOC lookup table
% %   - dOCV_SOC: dOCV-SOC lookup table
% %   - x_P: Model parameters
% %   - SOC_initial: Initial SoC
% %   - noise_std: Standard deviation of noise
% %   - noise_mean: Mean of noise
% 
% % Assign Data
% t = Data(:, 1);  
% I = Data(:, 2);  
% V = Data(:, 3);  
% n = length(I);
% 
% % Add biased noise to current signal
% noise = noise_std * randn(size(I)) + noise_mean;
% I = I + noise;
% 
% % Model Parameters
% R0 = x_P(1); R1 = x_P(2); C1 = x_P(3);
% capacity = 2;  
% 
% % Kalman Initialization
% V1_0 = 0;                     
% SOC_0 = SOC_initial;  
% x_hat_0 = [V1_0; SOC_0];  
% 
% P0 = diag([5e-5, 5e-2]);  
% Q = diag([1e-6, 1e-5]);    
% R = 6e-2;                         
% 
% x_hat_plus = zeros(2, n);   
% K = zeros(2, n);                
% 
% % EKF Loop
% for i = 1:n
%     if i == 1; dt = t(i);
%     else; dt = t(i)-t(i-1); end
% 
%     % State transition and input matrices
%     A = [exp(-dt / (R1 * C1)), 0; 0, 1];
%     B = [R1 * (1 - exp(-dt / (R1 * C1))); dt / (capacity * 3600)];
% 
%     % Prediction step
%     if i == 1 
%         x_hat_minus = A*x_hat_0 + B*I(i);    
%         P_minus = A*P0*A' + Q;                    
%     else
%         x_hat_minus = A*x_hat_plus(:,i-1) + B*I(i);  
%         P_minus = A*P_plus*A' + Q;                         
%     end
% 
%     % Interpolate OCV and dOCV
%     SOC = x_hat_minus(2);
%     OCV = interp1(OCV_SOC_25C(:, 1) / 100, OCV_SOC_25C(:, 2), SOC, 'spline');
%     dOCV = interp1(dOCV_SOC(:, 1) / 100, dOCV_SOC(:, 2), SOC, 'spline');
% 
%     C = [1, dOCV];
% 
%     % Kalman gain
%     K(:, i) = P_minus * C' / (C * P_minus * C' + R);
% 
%     % Measurement model    
%     y_est = OCV + R0 * I(i) + x_hat_minus(1);
% 
%     % Measurement update
%     err = V(i) - y_est;     
%     x_hat_plus(:, i) = x_hat_minus + K(:, i) * err;   
%     P_plus = (eye(2) - K(:, i) * C) * P_minus;            
% end
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% chnage in P0 %%%%%%%%%%%%%%%%%%%%%%%%%
% Load and Read Data
% [D_FUDS, D_HDS, D_BJDST] = Read_dynamic_data();
% load('OCV_SOC_relation.mat');   % OCV-SOC look-up table (25°C)
% dOCV_SOC = dOCV_SOC();          % dOCV-SOC look-up table
% 
% % Compute SOC using Coulomb counting (experimental results)
% [SOC_FUDS, SOC_HDS, SOC_BJDST] = SOC_measured(D_FUDS, D_HDS, D_BJDST);   % Effective range: 10-80%
% 
% % Model parameters (1RC model)
% x_P = [0.070248, 0.009953, 885.996888];  % PSO 1
% 
% % Define different P0 values
% P0_values = {diag([5e-5, 1]), diag([5e-5, 5e-1]), diag([5e-5, 5e-2]), diag([5e-5, 5e-3]), diag([5e-5, 5e-4])};
% 
% % Fixed Initial SoC, noise standard deviation, and noise mean
% SOC_initial = 0.7;  
% noise_std = 0.02;  
% noise_mean = 0.0;  
% 
% % Select dataset
% SOCdata = SOC_FUDS;       
% D = D_FUDS;      
% 
% % Time vector
% t = SOCdata(:, 1);           
% SOC_measure = SOCdata(:, 2);    
% 
% % Initialize figure
% figure; hold on; grid on;
% 
% % Loop through different P0 values
% for idx = 1:length(P0_values)
%     P0 = P0_values{idx};
% 
%     % Run EKF SOC estimation with different P0 values
%     [x_hat_plus] = SOC_model_EKF(D, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std, noise_mean, P0);
%     SOC_model = x_hat_plus(2, :)' * 100;  
% 
%     % Plot results
%     plot(t, SOC_model, 'LineWidth', 2, 'DisplayName', ['P0 [' num2str(P0(1,1)) ', ' num2str(P0(2,2)) ']']);
% end
% 
% % Plot measured SOC for reference
% plot(t, SOC_measure, 'k--', 'LineWidth', 2, 'DisplayName', 'Measured SOC');
% 
% % Format plot
% xlabel('Time (s)', 'FontSize', 18);
% ylabel('SOC (%)', 'FontSize', 18);
% title('Effect of Initial Covariance P0 on SoC Estimation (EKF)', 'FontSize', 18);
% legend('show', 'FontSize', 15);
% set(gcf, 'Color', 'w');
% set(gca, 'FontSize', 15);
% 
% %% Updated SOC_EKF function
% 
% function [x_hat_plus] = SOC_model_EKF(Data, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std, noise_mean, P0)
% % Extended Kalman Filter to predict SOC of a Lithium-ion cell.
% % Inputs:
% %   - Data: Driving profile data
% %   - OCV_SOC_25C: OCV-SOC lookup table
% %   - dOCV_SOC: dOCV-SOC lookup table
% %   - x_P: Model parameters
% %   - SOC_initial: Initial SoC
% %   - noise_std: Standard deviation of noise
% %   - noise_mean: Mean of noise
% %   - P0: Initial estimation error covariance
% 
% % Assign Data
% t = Data(:, 1);  
% I = Data(:, 2);  
% V = Data(:, 3);  
% n = length(I);
% 
% % Add biased noise to current signal
% noise = noise_std * randn(size(I)) + noise_mean;
% I = I + noise;
% 
% % Model Parameters
% R0 = x_P(1); R1 = x_P(2); C1 = x_P(3);
% capacity = 2;  
% 
% % Kalman Initialization
% V1_0 = 0;                     
% SOC_0 = SOC_initial;  
% x_hat_0 = [V1_0; SOC_0];  
% 
% Q = diag([1e-6, 1e-5]);    
% R = 6e-2;                         
% 
% x_hat_plus = zeros(2, n);   
% K = zeros(2, n);                
% 
% % EKF Loop
% for i = 1:n
%     if i == 1; dt = t(i);
%     else; dt = t(i)-t(i-1); end
% 
%     % State transition and input matrices
%     A = [exp(-dt / (R1 * C1)), 0; 0, 1];
%     B = [R1 * (1 - exp(-dt / (R1 * C1))); dt / (capacity * 3600)];
% 
%     % Prediction step
%     if i == 1 
%         x_hat_minus = A*x_hat_0 + B*I(i);    
%         P_minus = A*P0*A' + Q;                    
%     else
%         x_hat_minus = A*x_hat_plus(:,i-1) + B*I(i);  
%         P_minus = A*P_plus*A' + Q;                         
%     end
% 
%     % Interpolate OCV and dOCV
%     SOC = x_hat_minus(2);
%     OCV = interp1(OCV_SOC_25C(:, 1) / 100, OCV_SOC_25C(:, 2), SOC, 'spline');
%     dOCV = interp1(dOCV_SOC(:, 1) / 100, dOCV_SOC(:, 2), SOC, 'spline');
% 
%     C = [1, dOCV];
% 
%     % Kalman gain
%     K(:, i) = P_minus * C' / (C * P_minus * C' + R);
% 
%     % Measurement model    
%     y_est = OCV + R0 * I(i) + x_hat_minus(1);
% 
%     % Measurement update
%     err = V(i) - y_est;     
%     x_hat_plus(:, i) = x_hat_minus + K(:, i) * err;   
%     P_plus = (eye(2) - K(:, i) * C) * P_minus;            
% end
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% change in Q %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% [D_FUDS, D_HDS, D_BJDST] = Read_dynamic_data();
% load('OCV_SOC_relation.mat');   % OCV-SOC look-up table (25°C)
% dOCV_SOC = dOCV_SOC();          % dOCV-SOC look-up table
% 
% % Compute SOC using Coulomb counting (experimental results)
% [SOC_FUDS, SOC_HDS, SOC_BJDST] = SOC_measured(D_FUDS, D_HDS, D_BJDST);   % Effective range: 10-80%
% 
% % Model parameters (1RC model)
% x_P = [0.070248, 0.009953, 885.996888];  % PSO 1
% 
% % Define different P0 values
% Q_values = {diag([1e-4, 1e-5]), diag([1e-5, 1e-5]), diag([1e-6, 1e-5]), diag([1e-7, 1e-5]), diag([1e-8, 1e-5])};
% 
% % Fixed Initial SoC, noise standard deviation, and noise mean
% SOC_initial = 0.7;  
% noise_std = 0.02;  
% noise_mean = 0.0;  
% 
% % Select dataset
% SOCdata = SOC_FUDS;       
% D = D_FUDS;      
% 
% % Time vector
% t = SOCdata(:, 1);           
% SOC_measure = SOCdata(:, 2);    
% 
% % Initialize figure
% figure; hold on; grid on;
% 
% % Loop through different P0 values
% for idx = 1:length(Q_values)
%     Q = Q_values{idx};
% 
%     % Run EKF SOC estimation with different P0 values
%     [x_hat_plus] = SOC_model_EKF(D, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std, noise_mean, Q);
%     SOC_model = x_hat_plus(2, :)' * 100;  
% 
%     % Plot results
%     plot(t, SOC_model, 'LineWidth', 2, 'DisplayName', ['Q [' num2str(Q(1,1)) ', ' num2str(Q(2,2)) ']']);
% end
% 
% % Plot measured SOC for reference
% plot(t, SOC_measure, 'k--', 'LineWidth', 2, 'DisplayName', 'Measured SOC');
% 
% % Format plot
% xlabel('Time (s)', 'FontSize', 18);
% ylabel('SOC (%)', 'FontSize', 18);
% title('Effect of Initial Covariance Q on SoC Estimation (EKF)', 'FontSize', 18);
% legend('show', 'FontSize', 15);
% set(gcf, 'Color', 'w');
% set(gca, 'FontSize', 15);
% 
% %% Updated SOC_EKF function
% 
% function [x_hat_plus] = SOC_model_EKF(Data, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std, noise_mean, Q)
% % Extended Kalman Filter to predict SOC of a Lithium-ion cell.
% % Inputs:
% %   - Data: Driving profile data
% %   - OCV_SOC_25C: OCV-SOC lookup table
% %   - dOCV_SOC: dOCV-SOC lookup table
% %   - x_P: Model parameters
% %   - SOC_initial: Initial SoC
% %   - noise_std: Standard deviation of noise
% %   - noise_mean: Mean of noise
% %   - P0: Initial estimation error covariance
% 
% % Assign Data
% t = Data(:, 1);  
% I = Data(:, 2);  
% V = Data(:, 3);  
% n = length(I);
% 
% % Add biased noise to current signal
% noise = noise_std * randn(size(I)) + noise_mean;
% I = I + noise;
% 
% % Model Parameters
% R0 = x_P(1); R1 = x_P(2); C1 = x_P(3);
% capacity = 2;  
% 
% % Kalman Initialization
% V1_0 = 0;                     
% SOC_0 = SOC_initial;  
% x_hat_0 = [V1_0; SOC_0];  
% 
% P0 = diag([5e-5, 5e-2]);
% %Q = diag([1e-6, 1e-5]);    
% R = 6e-2;                         
% 
% x_hat_plus = zeros(2, n);   
% K = zeros(2, n);                
% 
% % EKF Loop
% for i = 1:n
%     if i == 1; dt = t(i);
%     else; dt = t(i)-t(i-1); end
% 
%     % State transition and input matrices
%     A = [exp(-dt / (R1 * C1)), 0; 0, 1];
%     B = [R1 * (1 - exp(-dt / (R1 * C1))); dt / (capacity * 3600)];
% 
%     % Prediction step
%     if i == 1 
%         x_hat_minus = A*x_hat_0 + B*I(i);    
%         P_minus = A*P0*A' + Q;                    
%     else
%         x_hat_minus = A*x_hat_plus(:,i-1) + B*I(i);  
%         P_minus = A*P_plus*A' + Q;                         
%     end
% 
%     % Interpolate OCV and dOCV
%     SOC = x_hat_minus(2);
%     OCV = interp1(OCV_SOC_25C(:, 1) / 100, OCV_SOC_25C(:, 2), SOC, 'spline');
%     dOCV = interp1(dOCV_SOC(:, 1) / 100, dOCV_SOC(:, 2), SOC, 'spline');
% 
%     C = [1, dOCV];
% 
%     % Kalman gain
%     K(:, i) = P_minus * C' / (C * P_minus * C' + R);
% 
%     % Measurement model    
%     y_est = OCV + R0 * I(i) + x_hat_minus(1);
% 
%     % Measurement update
%     err = V(i) - y_est;     
%     x_hat_plus(:, i) = x_hat_minus + K(:, i) * err;   
%     P_plus = (eye(2) - K(:, i) * C) * P_minus;            
% end
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% chnages in R0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% [D_FUDS, D_HDS, D_BJDST] = Read_dynamic_data();
% load('OCV_SOC_relation.mat');   % OCV-SOC look-up table (25°C)
% dOCV_SOC = dOCV_SOC();          % dOCV-SOC look-up table
% 
% % Compute SOC using Coulomb counting (experimental results)
% [SOC_FUDS, SOC_HDS, SOC_BJDST] = SOC_measured(D_FUDS, D_HDS, D_BJDST);   % Effective range: 10-80%
% 
% % Model parameters (1RC model)
% x_P = [0.070248, 0.009953, 885.996888];  % PSO 1
% 
% % Define different R values
% R_values = [6, 6e-1, 6e-2, 6e-3, 6e-4];
% 
% % Fixed Initial SoC, noise standard deviation, and noise mean
% SOC_initial = 0.7;  
% noise_std = 0.02;  
% noise_mean = 0.0;  
% 
% % Select dataset
% SOCdata = SOC_FUDS;       
% D = D_FUDS;      
% 
% % Time vector
% t = SOCdata(:, 1);           
% SOC_measure = SOCdata(:, 2);    
% 
% % Initialize figure
% figure; hold on; grid on;
% 
% % Loop through different P0 values
% for idx = 1:length(R_values)
%     R = R_values(idx);
% 
%     % Run EKF SOC estimation with different P0 values
%     [x_hat_plus] = SOC_model_EKF(D, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std, noise_mean, R);
%     SOC_model = x_hat_plus(2, :)' * 100;  
% 
%     % Plot results
%     plot(t, SOC_model, 'LineWidth', 2, 'DisplayName', ['R [' num2str(R) ']']);
% end
% 
% % Plot measured SOC for reference
% plot(t, SOC_measure, 'k--', 'LineWidth', 2, 'DisplayName', 'Measured SOC');
% 
% % Format plot
% xlabel('Time (s)', 'FontSize', 18);
% ylabel('SOC (%)', 'FontSize', 18);
% title('Effect of Initial Covariance Q on SoC Estimation (EKF)', 'FontSize', 18);
% legend('show', 'FontSize', 15);
% set(gcf, 'Color', 'w');
% set(gca, 'FontSize', 15);
% 
% %% Updated SOC_EKF function
% 
% function [x_hat_plus] = SOC_model_EKF(Data, OCV_SOC_25C, dOCV_SOC, x_P, SOC_initial, noise_std, noise_mean, R)
% Extended Kalman Filter to predict SOC of a Lithium-ion cell.
% Inputs:
%   - Data: Driving profile data
%   - OCV_SOC_25C: OCV-SOC lookup table
%   - dOCV_SOC: dOCV-SOC lookup table
%   - x_P: Model parameters
%   - SOC_initial: Initial SoC
%   - noise_std: Standard deviation of noise
%   - noise_mean: Mean of noise
%   - P0: Initial estimation error covariance

% % Assign Data
% t = Data(:, 1);  
% I = Data(:, 2);  
% V = Data(:, 3);  
% n = length(I);
% 
% % Add biased noise to current signal
% noise = noise_std * randn(size(I)) + noise_mean;
% I = I + noise;
% 
% % Model Parameters
% R0 = x_P(1); R1 = x_P(2); C1 = x_P(3);
% capacity = 2;  
% 
% % Kalman Initialization
% V1_0 = 0;                     
% SOC_0 = SOC_initial;  
% x_hat_0 = [V1_0; SOC_0];  
% 
% P0 = diag([5e-5, 5e-2]);
% Q = diag([1e-6, 1e-5]);    
% %R = 6e-2;                         
% 
% x_hat_plus = zeros(2, n);   
% K = zeros(2, n);                
% 
% % EKF Loop
% for i = 1:n
%     if i == 1; dt = t(i);
%     else; dt = t(i)-t(i-1); end
% 
%     % State transition and input matrices
%     A = [exp(-dt / (R1 * C1)), 0; 0, 1];
%     B = [R1 * (1 - exp(-dt / (R1 * C1))); dt / (capacity * 3600)];
% 
%     % Prediction step
%     if i == 1 
%         x_hat_minus = A*x_hat_0 + B*I(i);    
%         P_minus = A*P0*A' + Q;                    
%     else
%         x_hat_minus = A*x_hat_plus(:,i-1) + B*I(i);  
%         P_minus = A*P_plus*A' + Q;                         
%     end
% 
%     % Interpolate OCV and dOCV
%     SOC = x_hat_minus(2);
%     OCV = interp1(OCV_SOC_25C(:, 1) / 100, OCV_SOC_25C(:, 2), SOC, 'spline');
%     dOCV = interp1(dOCV_SOC(:, 1) / 100, dOCV_SOC(:, 2), SOC, 'spline');
% 
%     C = [1, dOCV];
% 
%     % Kalman gain
%     K(:, i) = P_minus * C' / (C * P_minus * C' + R);
% 
%     % Measurement model    
%     y_est = OCV + R0 * I(i) + x_hat_minus(1);
% 
%     % Measurement update
%     err = V(i) - y_est;     
%     x_hat_plus(:, i) = x_hat_minus + K(:, i) * err;   
%     P_plus = (eye(2) - K(:, i) * C) * P_minus;            
% end
% 
% end