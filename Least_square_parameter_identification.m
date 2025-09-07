clear all
load ECM2
%load ECM2_new

%%%%% Q) a,b %%%%%
% Fit polynomial model to OCV data
Pocv = polyfit(OCV(:,1),OCV(:,2),10);

% Open loop estimation of z
Ts = 0.1;           % Sampling time (s)
Q  = 60*3600;       % Capacity (As)
z(1) = 0.7;                                 % Initial SoC
vOC(1) = polyval(Pocv,z(1));                % Inital OCV
for k = 1:length(Time)-1 
    z(k+1) = z(k)+(Ts/Q)*Current(k);        % Simulated SoC
    vOC(k+1) = polyval(Pocv,z(k+1));        % Corresponding OCV
end

% Standard LSQ regression model
N   = length(Time);
Y   = Voltage - vOC';
Phi = [Y(1:N-1) Current(2:N) Current(1:N-1)];

theta = inv(Phi'*Phi)*Phi'*Y(2:N);

% Simple check
alpha_hat = theta(1)
alpha     = exp(-Ts/(R1*C))


R0_hat = theta(2)
R0

R1_hat = (theta(3)+theta(2)*theta(1))/(1-theta(1))
R1

% C_hat  = -1/(R1_hat*log(alpha_hat))
C_hat  = (-1/(R1_hat*log(alpha_hat)))/10

v1 = zeros(1, length(Time));
v = zeros(1, length(Time));


%%%%%%%%%%%%%%%%%%%%%
% Discrete-time model computation
for k = 1:length(Time)-1
    % Update v1 using estimated parameters
    v1(k+1) = alpha_hat * v1(k) + R1_hat*(1 - alpha_hat) * Current(k); % * Ts / C_hat;

    % Compute the output voltage
    v(k+1) = vOC(k+1) + R0_hat * Current(k+1) + v1(k+1);
end

V_pred=v';

figure(1);
plot(Time, Voltage, 'b', 'LineWidth', 1.5);
hold on;
plot(Time, V_pred, 'r--', 'LineWidth', 1.5);
ylim([3.5 4.2])
xlabel('Time (s)');
ylabel('Voltage (V)');
legend('Measured Voltage', 'Predicted Voltage with 1st order RC ');
title('Comparison of Measured and Predicted Voltage');
grid on;
hold off;

% Calculate the error between real and predicted voltage
error_r = Voltage - V_pred;

% Square the errors
squared_error_r = error_r .^ 2;

% Compute the mean of the squared errors (MSE)
MSE_r = mean(squared_error_r);

% Calculate the root mean squared error (RMSE)
RMSE_r = sqrt(MSE_r);

% Display the RMSE value
fprintf('RMSE: %f\n', RMSE_r);

figure(2)
plot(error_r)
ylim([-0.0010 0.0006])
ylabel("voltage difference bet T-voltage and pred-voltage")
xlabel("Time")
title("Error")

figure(3)
plot(Current)
ylabel("Current")
xlabel("Time")
title("Current vs Time")

figure(4)
plot(Voltage)
ylabel("Voltage")
xlabel("Time")
title("Voltage vs Time")