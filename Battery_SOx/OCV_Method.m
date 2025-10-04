%% SoC estimation using OCV method
clc; clear;

% Determine OCV-SOC Relationship
OCV_SOC_25C = OCV_SOC(); 
% OCV_SOC_25C represents the relationship between OCV and SOC at 25°C, simulating normal temperature conditions.

% Load Test Data and Preprocess
load('DST_80SOC_25C.mat');  % Load test data
t = num(:,2);         % Test time (s)
I = num(:,7);         % Current (A): positive for charge, negative for discharge
V = num(:,8);       % Terminal voltage (V)
Dc = num(:,10);   % Discharge capacity (Ah)

% Clear unnecessary variables
clear num txt;

% Plot the data 
figure;
subplot(2, 1, 1)
plot(t/3600, I, 'LineWidth', 2)
xlabel('Time (h)', 'FontSize', 18); 
ylabel('Current (A)', 'FontSize', 18); 
set(gca, 'FontSize', 15);
xlim ([0 9])

subplot(2, 1, 2)
plot(t/3600, V, 'LineWidth', 2)
xlabel('Time (h)', 'FontSize', 18) 
ylabel('Voltage (V)', 'FontSize', 18)
xlim ([0 9])

set(gcf, 'Color', 'w'); 
grid on;                            % Enable grid
set(gca, 'FontSize', 15);  % Set axis font size

% Apply the OCV method
% Select data from the start index, the data after fully charged
ST = 333;
I = I(ST:end);
V = V(ST:end);
t = t(ST:end) - t(ST); % Adjust time to start from 0
SOC_0 = 100;          % Initial SoC
Q = 2;                       % Cell capacity
R0 = 0.0735;          % Ohmic Resistance, obtained from pulse test 
%R0 = 0.08191;

% Initialize variables
%%%%%%%%%%%%%%%%
n = length(I);
SOC = zeros(n, 1);
SOC(1) = SOC_0;

% Coulomb counting
for i = 1:n-1
    dt = t(i+1) - t(i); % Time step
    SOC(i+1) = SOC(i) + (dt * I(i) / (Q * 3600)) * 100;   % measured/reference SoC
end
%%%% true SOC
%%%%%%%%%%%%%%%%%%%%%%

% SOC-OCV interpolation setup
SOC_points = OCV_SOC_25C(:, 1); % SOC sample points
OCV_values = OCV_SOC_25C(:, 2); % Corresponding OCV values


% OCV Method Simulate
for i = 1:n  

    % OCV calculation based on the model
    OCV(i) = V(i) - R0 * I(i);

    % Interpolate SoC from OCV-SOC relationship
    SoC_est(i) = interp1(OCV_values, SOC_points,  OCV(i), 'linear','extrap');

end


% Plot the result
figure; hold on

plot(t/3600, SoC_est, 'LineWidth', 2, 'DisplayName', 'SoC estimate');
plot(t/3600, SOC, 'LineWidth', 3, 'DisplayName', 'True');

xlabel('Time (h)', 'FontSize', 18); 
ylabel('SoC (%)', 'FontSize', 18); 
set(gca, 'FontSize', 15);
 % xlim ([4 8])
set(gcf, 'Color', 'w'); 
grid on;  % Enable grid
set(gca, 'FontSize', 15); % Set axis font size
legend('show', 'FontSize', 15);


figure; hold on

plot(t/3600, V, 'LineWidth', 2, 'DisplayName', 'Terminal voltage');
plot(t/3600, OCV, 'LineWidth', 2, 'DisplayName', 'OCV');

xlabel('Time (h)', 'FontSize', 18); 
ylabel('Voltage (V)', 'FontSize', 18); 
set(gca, 'FontSize', 15);
 % xlim ([4 8])
set(gcf, 'Color', 'w'); 
grid on;  % Enable grid
set(gca, 'FontSize', 15); % Set axis font size
legend('show', 'FontSize', 15);

%%
%%%%%% RO estimation methods%%%%%%

% Fit polynomial model to OCV data
Pocv = polyfit(OCV_SOC_25C(:,1),OCV_SOC_25C(:,2),8);

% OCV_OCV= OCV_SOC_25C(:,2);
% OCV_SOC= OCV_SOC_25C(:,1);

% Open loop estimation of z
Ts = 0.1;           % Sampling time (s)
Q  = 2*3600;       % Capacity (As)
z(1) = 100;                                 % Initial SoC
vOC(1) = polyval(Pocv,z(1));                % Inital OCV
for k = 1:length(t)-1
    z(k+1) = z(k)+(Ts/Q)*I(k);        % Simulated SoC
    vOC(k+1) = polyval(Pocv,z(k+1));        % Corresponding OCV
end

%v0 = Voltage - vOC';

v0 = V - vOC';
%% Least square fit Method-1
Y   = v0;
Phi = I;
R0hat = inv(Phi'*Phi)*Phi'*Y

% figure(3)
% plot(I,v0,'.',I,I*R0hat,'r')
% xlabel('Current'),ylabel('\it v_0')
R0hat = R0hat(end);

% theta_ext = inv(Phi'*Phi)*Phi'*Y;
% R0_ext_1 = theta_ext(1);  % The coefficient for Current
%%  least squares for Method (regresion model) Method-2
% comparison plot between estimated voltage from R0_hat of regresion vs measured voltage
N = length(t);
Y = v0(2:N)-v0(1:N-1);
Phi = [I(2:N) -I(1:N-1)];

theta = inv(Phi'*Phi)*Phi'*Y;
% Assuming theta(1) and theta(2) are the same:
R0_estimated = theta(1);  % or theta(2)

% Predicted voltage using the Rint model
V_predicted = vOC(k) - R0_estimated * I;

% voc_n=vOC';
% % Plot to compare predicted and actual voltages
% figure(1);
% plot(Time, Voltage, 'b', 'LineWidth', 1.5); % Actual voltage
% hold on;
% plot(Time, V_predicted, 'r--', 'LineWidth', 1.5); % Predicted voltage
% xlabel('Time (s)');
% ylabel('Voltage (V)');
% legend('Measured Voltage', 'Predicted Voltage');
% title('Comparison of Measured and Predicted Voltages');
% grid on;
% hold off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% RLS algorithm _normal  model
% applying RLS algo to theta obtained from regreesion model 
% comparison plot between estimated voltage from R0_hat of regresion with RLS method vs measured voltage
Y   = v0;
Phi = I;
lambda = 1     % Try different values of lambda (forgetting factor) 

theta(1) = 0.01;   % Initial parameter guess
P(1)     = 1;    % Initialization of P
for k = 2:length(t)
    eps      = Y(k,:) - Phi(k,:)*theta(k-1);
    K        = P(k-1)*Phi(k,:)/(lambda+Phi(k,:)*P(k-1)*Phi(k,:)');
    theta(k) = theta(k-1) + K*eps;
    P(k)     = (1/lambda)*(1-K*Phi(k,:))*P(k-1);
end

% Extract the final estimate for R0
R0_hat = theta(end);

% Display the estimated R0
fprintf('Estimated R0: %f\n', R0_hat);









%% RLS Algorithm - Regression Model
% Applying RLS to estimate theta from the regression model
% Comparison plot between estimated voltage using R0_hat from regression with RLS vs measured voltage

Y   = v0(2:N) - v0(1:N-1);   % Output difference (size: (N-1) × 1)
Phi = [I(2:N), -I(1:N-1)];   % Regression matrix (size: (N-1) × 2)

lambda = 1;  % Forgetting factor

% Initialize parameters
theta = zeros(2, N-1);  % Now it's a 2-row vector (since Phi has 2 columns)
P = eye(2);  % Initial covariance matrix

% RLS Algorithm
for k = 2:N-1  % Loop over valid indices only
    Phi_k = Phi(k, :)';  % Column vector of regression terms
    eps = Y(k) - Phi_k' * theta(:, k-1);  % Prediction error
    K = P * Phi_k / (lambda + Phi_k' * P * Phi_k);  % Gain computation
    theta(:, k) = theta(:, k-1) + K * eps;  % Update theta
    P = (P - K * Phi_k' * P) / lambda;  % Update covariance matrix
end

% Extract the final estimate for R0
R0_hat = theta(1, end);

% Display the estimated R0
fprintf('Estimated R0: %f\n', R0_hat);



