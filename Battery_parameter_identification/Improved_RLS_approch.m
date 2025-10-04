clear; clc; close all;

% --- Load data ---
if exist('ECM2.mat','file')
    load('ECM2.mat');  
else
    error('ECM2.mat not found.');
end
Time    = Time(:);
Voltage = Voltage(:);
Current = Current(:);
N       = length(Time);
Ts      = mean(diff(Time));
Qrated_Ah = 60;      
Q        = Qrated_Ah * 3600; 

% --- OCV polynomial fit ---
if size(OCV,2) < 2
    error('OCV must have columns [SOC OCV_voltage].');
end
Pocv = polyfit(OCV(:,1), OCV(:,2), 10);

% --- SOC & OCV calculation ---
z = zeros(N,1); 
z(1) = 0.7;  % initial SOC
Uoc = zeros(N,1); 
Uoc(1) = polyval(Pocv, z(1));
for k = 1:N-1
    z(k+1) = z(k) + (Ts/Q)*Current(k);   % discharge current convention
    z(k+1) = min(max(z(k+1),0),1);       % keep in [0,1]
    Uoc(k+1) = polyval(Pocv, z(k+1));
end

% --- Initialization ---
theta_hat = zeros(5,1);        % Initial parameter vector
P = (0.1)^2*eye(5);            % Initial covariance matrix
lambda = 0.99;                 % Forgetting factor
theta_hist = zeros(5,N);       % Store theta estimates
params = zeros(7,N);           % Store physical params: [R0, R1, R2, C1, C2, tau1, tau2]
out = Uoc - Voltage;           % Residual definition
T = Ts;                        % Sampling period

% --- Recursive RLS estimation ---
for k = 3:N
    % Information vector
    phi = [out(k-1);
           out(k-2);
           Current(k);
           Current(k-1);
           Current(k-2)];
       
    yk = out(k);   % measured output
    
    % Gain vector
    K = (P * phi) / (lambda + phi' * P * phi);
    
    % Prediction
    y_pred = phi' * theta_hat;
    
    % Parameter update
    theta_hat = theta_hat + K * (yk - y_pred);
    theta_hist(:,k) = theta_hat;
    
    % Covariance update
    P = (1/lambda) * (P - K * phi' * P);
    
    % --- Map ARX -> Physical Parameters ---
    k1 = theta_hat(1); k2 = theta_hat(2); k3 = theta_hat(3);
    k4 = theta_hat(4); k5 = theta_hat(5);
    
    den = k1 + k2 - 1;
    if abs(den) < 1e-12, den = sign(den)*1e-12; end
    
    k0 = T^2 / den;
    a  = k2 * k0;
    b  = -k0 * (k1 + 2*k2) / T;
    c  = k0 * (k3 + k4 + k5) / T^2;
    d  = -k0 * (k4 + 2*k5) / T;
    R0 = k5 / max(k2,1e-12);  % avoid div by zero
    
    Delta = max(b^2 - 4*a, 0);
    root1 = (b + sqrt(Delta)) / 2;
    root2 = (b - sqrt(Delta)) / 2;
    tau1 = min(root1, root2);
    tau2 = max(root1, root2);
    
    % Avoid division by near-zero
    R2 = (tau2*c + R0*b - tau2*R0 - d) / max(tau2 - tau1, 1e-12);
    R1 = max(c - R2 - R0, 1e-12);
    R2 = max(R2, 1e-12);
    
    C1 = abs(tau1) / R1;
    C2 = abs(tau2) / R2;
    
    % Store physical parameters
    params(:,k) = [R0; R1; R2; C1; C2; tau1; tau2];
end

% --- Post-processing ---

% Predict output using ARX coefficients
out_pred = zeros(N,1);
out_pred(1:2) = out(1:2);
for k = 3:N
    k1 = theta_hist(1,k); k2 = theta_hist(2,k); k3 = theta_hist(3,k);
    k4 = theta_hist(4,k); k5 = theta_hist(5,k);
    out_pred(k) = k1*out(k-1) + k2*out(k-2) + ...
                  k3*Current(k) + k4*Current(k-1) + k5*Current(k-2);
end

% Simulate 2-RC ECM with final estimated parameters
v1 = zeros(1, N);
v2 = zeros(1, N);
v = zeros(1, N);
R0 = params(1,end);
R1 = params(2,end);
R2 = params(3,end);
C1 = params(4,end);
C2 = params(5,end);
A1 = exp(-T/(R1*C1));
A2 = exp(-T/(R2*C2));

fprintf('Final parameters at last time step:\n');
fprintf('A1 = %.6g, A2 = %.6g, R1 = %.6g, R2 = %.6g, R0 = %.6g\n', A1, A2, R1, R2, R0);

for k = 1:N-1
    v1(k+1) = A1 * v1(k) + R1 * (1 - A1) * Current(k);
    v2(k+1) = A2 * v2(k) + R2 * (1 - A2) * Current(k);
    v(k+1)  = Uoc(k+1) + R0 * Current(k+1) + v1(k+1) + v2(k+1);
end

% --- Plot results ---
figure('Name','RLS ARX Parameter Evolution');
plot(Time, theta_hist','LineWidth',1.2);
xlabel('Time (s)'); ylabel('Parameter Value');
legend('k1','k2','k3','k4','k5');
title('Strict RLS Parameter Evolution'); grid on;

figure('Name','Estimated Physical Parameters');
plot(Time, params(1,:), '-r', Time, params(2,:), '-g', Time, params(3,:), '-b', ...
     Time, params(4,:), '-m', Time, params(5,:), '-c', 'LineWidth', 1.1);
xlabel('Time (s)'); ylabel('Parameter Value');
legend('R0 (\Omega)','R1 (\Omega)','R2 (\Omega)','C1 (F)','C2 (F)');
title('Estimated Physical Battery Parameters Over Time'); grid on;

figure('Name','Measured vs Predicted Output');
plot(Time, out, 'b', 'LineWidth',1); hold on;
plot(Time, out_pred, 'r--', 'LineWidth',1.2);
plot(Time, Uoc'-v, 'g-.', 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('U_{oc} - U_L (V)');
legend('Measured','RLS ARX Predicted','Physical Param Predicted');
title('Output Comparison'); grid on;

figure('Name','Terminal Voltage Comparison');
plot(Time, Voltage, 'b', 'LineWidth',1); hold on;
plot(Time, Uoc - out_pred, 'r--', 'LineWidth',1.2);
plot(Time, v, 'g-.', 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('Terminal Voltage (V)');
ylim([3.6 4.2]);
legend('Measured Voltage','RLS ARX Predicted','Physical Param Predicted');
title('Terminal Voltage Comparison'); grid on;

rmse_val_1 = sqrt(mean(((Uoc - out_pred) - Voltage).^2));
fprintf('RMSE of Measured vs Parameter-based Predicted Voltage = %.6f V\n', rmse_val_1);

rmse_val_2 = sqrt(mean((v' - Voltage).^2));
fprintf('RMSE of Measured vs ECM_predicted_parameter = %.6f V\n', rmse_val_2);