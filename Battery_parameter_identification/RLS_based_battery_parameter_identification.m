clear all
%load ECM1 
load ECM1
% Fit polynomial model to OCV data
Pocv = polyfit(OCV(:,1),OCV(:,2),10);

% Open loop estimation of z
Ts = 0.1;           % Sampling time (s)
Q  = 60*3600;       % Capacity (As)
z(1) = 0.5;                                 % Initial SoC
vOC(1) = polyval(Pocv,z(1));                % Inital OCV
for k = 1:length(Time)-1
    z(k+1) = z(k)+(Ts/Q)*Current(k);        % Simulated SoC
    vOC(k+1) = polyval(Pocv,z(k+1));        % Corresponding OCV
end

%v0 = Voltage - vOC';

%%%%%%%%%%%%%%%%%%%%
for k = 1:length(Time)    
    v0 = Voltage(k) - vOC';
    R_not = v0/Current(k);
   
end
%%%%%%%%%%%%%%%%%%%%%%
v0 = Voltage - vOC';

%% voltage comparison for different lambda_values
% Given initial data
Y = v0;
Phi = Current;
lambda_values = [0.97, 0.98, 0.99]; % Different values of lambda (forgetting factor)
Time = 1:length(Y); % Assuming time is a vector of same length as Y

% Initialize storage for theta and predicted voltages
theta_all = zeros(length(lambda_values), length(Time));
V_pred_all = zeros(length(lambda_values), length(Time));

% Loop over each lambda value
for i = 1:length(lambda_values)
    lambda = lambda_values(i);

    % Initialize theta and P for RLS
    theta = zeros(1, length(Time));
    theta(1) = 0.01; % Initial parameter guess
    P = zeros(1, length(Time));
    P(1) = 1; % Initialization of P

    % RLS update loop
    for k = 2:length(Time)
        eps = Y(k,:) - Phi(k,:)*theta(k-1);
        K = P(k-1)*Phi(k,:)/(lambda + Phi(k,:)*P(k-1)*Phi(k,:)');
        theta(k) = theta(k-1) + K*eps;
        P(k) = (1/lambda) * (1 - K*Phi(k,:)) * P(k-1);
    end
    % Extract the final estimate for R0
    R0_hat = theta(end)

    % Initialize v_r
    v_r = zeros(1, length(Time));

    % Discrete-time model computation
    for k = 1:length(Time)-1
        % Compute the output voltage
        v_r(k+1) = vOC(k+1) + R0_hat * Current(k+1);
    end
    V_pred_r = v_r';
    
    % Store the predicted voltages
    V_pred_all(i,:) = V_pred_r;

    % Store theta values for this lambda
    theta_all(i,:) = theta;
end

% Plotting results
figure(6);
plot(Time, Voltage, 'b', 'LineWidth', 1.5);
ylim([2.75 4.5]) 
hold on;
plot(Time, V_pred_all(1,:), 'r', 'LineWidth', 1.5);
plot(Time, V_pred_all(2,:), 'g', 'LineWidth', 1.5);
plot(Time, V_pred_all(3,:), 'm', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Voltage (V)');
legend('Measured Voltage', 'Predicted Voltage with \lambda=0.97', 'Predicted Voltage with \lambda=0.98', 'Predicted Voltage with \lambda=0.99');
title('Comparison of Measured and Predicted Voltages with Different \lambda Values');
grid on;
hold off;

% Calculate the error between real and predicted voltage
error_r = Voltage - V_pred_all(3,:)';

% Square the errors
squared_error_r = error_r .^ 2;

% Compute the mean of the squared errors (MSE)
MSE_r = mean(squared_error_r);

% Calculate the root mean squared error (RMSE)
RMSE_r = sqrt(MSE_r);

% Display the RMSE value
fprintf('RMSE: %f\n', RMSE_r);

RMSE_all = zeros(length(lambda_values),1);

for i = 1:length(lambda_values)
    error_r = Voltage - V_pred_all(i,:)';
    RMSE_all(i) = sqrt(mean(error_r.^2));
end

disp(RMSE_all)
