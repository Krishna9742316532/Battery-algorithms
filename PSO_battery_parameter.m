%%
clc
clear
% Load Data
load ECM2
Ts = 0.1;           % Sampling time (s)
Q  = 60*3600;       % Capacity (As)

% Fit polynomial model to OCV data
Pocv = polyfit(OCV(:,1), OCV(:,2), 10);

% PSO Setup
n_particles = 50;
max_iter = 15;
c1 = 1.5; c2 = 1.5; w = 0.8;   % PSO parameters

% Parameter bounds: [R0, R1, C1]
lb = [0.001, 0.001, 10];
ub = [0.5,   0.5,   5000];

% Initialize swarm
rng(1);
X = lb' + (ub'-lb') .* rand(3, n_particles); % positions
V = randn(3, n_particles) * 0.01;            % velocities

% Evaluate initial particles
pbest = X;
pbest_obj = arrayfun(@(i) objective(X(:,i), Time, Current, Voltage, Pocv, Ts, Q), 1:n_particles);
[gbest_obj, idx] = min(pbest_obj);
gbest = pbest(:, idx);

% PSO Loop
for iter = 1:max_iter
    r1 = rand(3, n_particles);
    r2 = rand(3, n_particles);

    % Update velocity and position
    V = w*V + c1*r1.*(pbest - X) + c2*r2.*(gbest - X);
    X = X + V;

    % Bound handling
    X = max(min(X, ub'), lb');

    % Evaluate
    obj = arrayfun(@(i) objective(X(:,i), Time, Current, Voltage, Pocv, Ts, Q), 1:n_particles);

    % Update pbest
    mask = obj < pbest_obj;
    pbest(:, mask) = X(:, mask);
    pbest_obj(mask) = obj(mask);

    % Update gbest
    [new_best, idx] = min(pbest_obj);
    if new_best < gbest_obj
        gbest_obj = new_best;
        gbest = pbest(:, idx);
    end

    fprintf('Iter %d, Best Obj = %.6f\n', iter, gbest_obj);
end

% Final Simulation with best parameters
[~, v_pred] = objective(gbest, Time, Current, Voltage, Pocv, Ts, Q);

figure;
plot(Time, Voltage, 'b', 'LineWidth', 1.5); hold on;
plot(Time, v_pred, 'r--', 'LineWidth', 1.5);
legend('Measured Voltage', 'Predicted Voltage');
xlabel('Time [s]'); ylabel('Voltage [V]');
title('True vs Predicted Voltage (PSO)');
grid on;

fprintf('Optimal Parameters: R0=%.5f, R1=%.5f, C1=%.2f\n', gbest(1), gbest(2), gbest(3));

function [J, v_sim] = objective(params, Time, Current, Voltage, Pocv, Ts, Q)
    R0 = params(1); R1 = params(2); C1 = params(3);

    % Initialize states
    z = 0.7;              % initial SOC
    vRC = 0;              % RC branch voltage
    v_sim = zeros(size(Time));

    for k = 1:length(Time)
        % Update SOC
        if k > 1
            z = z + (Ts/Q) * Current(k-1);
        end

        % OCV
        vOC = polyval(Pocv, z);

        % RC branch update (discrete form)
        alpha = exp(-Ts/(R1*C1));
        vRC = alpha*vRC + R1*(1-alpha)*Current(k);

        % Terminal voltage
        v_sim(k) = vOC + vRC + R0*Current(k);
    end

    % Cost function (MSE)
    J = mean((Voltage - v_sim).^2);
end

%%
clc
clear
load ECM2
Ts = 0.1;
Q  = 60*3600;
Pocv = polyfit(OCV(:,1), OCV(:,2), 10);

lb_values = [0.01, 0.001, 0.0001];
results = zeros(length(lb_values), length(lb_values)); % RMSEs
params_store = zeros(length(lb_values), length(lb_values), 3); % Store parameter vectors

for i = 1:length(lb_values)
    for j = 1:length(lb_values)
        lb_R0 = lb_values(i);
        lb_R1 = lb_values(j);
        lb = [lb_R0, lb_R1, 10];
        ub = [0.5, 0.5, 5000];
        n_particles = 50;
        max_iter = 15;
        c1 = 1.5; c2 = 1.5; w = 0.8;
        rng(1);
        X = lb' + (ub'-lb') .* rand(3, n_particles);
        V = randn(3, n_particles) * 0.01;
        pbest = X;
        pbest_obj = arrayfun(@(k) objective(X(:,k), Time, Current, Voltage, Pocv, Ts, Q), 1:n_particles);
        [gbest_obj, idx] = min(pbest_obj);
        gbest = pbest(:, idx);
        for iter = 1:max_iter
            r1 = rand(3, n_particles); r2 = rand(3, n_particles);
            V = w*V + c1*r1.*(pbest - X) + c2*r2.*(gbest - X);
            X = X + V;
            X = max(min(X, ub'), lb');
            obj = arrayfun(@(k) objective(X(:,k), Time, Current, Voltage, Pocv, Ts, Q), 1:n_particles);
            mask = obj < pbest_obj;
            pbest(:, mask) = X(:, mask);
            pbest_obj(mask) = obj(mask);
            [new_best, idx] = min(pbest_obj);
            if new_best < gbest_obj
                gbest_obj = new_best;
                gbest = pbest(:, idx);
            end
        end
        results(i, j) = gbest_obj; % Store RMSE
        params_store(i, j, :) = gbest; % Store optimal parameters
        fprintf('lb_R0 = %.4f, lb_R1 = %.4f, RMSE = %.6f, R0 = %.5f, R1 = %.5f, C1 = %.2f\n', ...
            lb_R0, lb_R1, gbest_obj, gbest(1), gbest(2), gbest(3));
    end
end

figure;
imagesc(log10(results)); 
xticks(1:length(lb_values)); xticklabels(arrayfun(@num2str, lb_values, 'UniformOutput', false));
yticks(1:length(lb_values)); yticklabels(arrayfun(@num2str, lb_values, 'UniformOutput', false));
xlabel('lb\_R1'); ylabel('lb\_R0');
colorbar;
title('Log_{10}(RMSE) over lb\_R0 and lb\_R1');


%%
clc;
clear;
% Load Data
load ECM2
Ts = 0.1;           % Sampling time (s)
Q  = 60*3600;        % Capacity (As)
Pocv = polyfit(OCV(:,1), OCV(:,2), 10);
lb_values = [0.01, 0.001, 0.0001];
results = zeros(length(lb_values), length(lb_values)); % RMSEs
params_store = zeros(length(lb_values), length(lb_values), 3); % Store parameter vectors
for i = 1:length(lb_values)
    for j = 1:length(lb_values)
        lb_R0 = lb_values(i);
        lb_R1 = lb_values(j);
        lb = [lb_R0, lb_R1, 10];
        ub = [0.5, 0.5, 5000];
        n_particles = 50;
        max_iter = 15;
        c1 = 1.5; c2 = 1.5; w = 0.8;
        rng(1);
        X = lb' + (ub'-lb') .* rand(3, n_particles);
        V = randn(3, n_particles) * 0.01;
        pbest = X;
        pbest_obj = arrayfun(@(k) objective(X(:,k), Time, Current, Voltage, Pocv, Ts, Q), 1:n_particles);
        [gbest_obj, idx] = min(pbest_obj);
        gbest = pbest(:, idx);
        for iter = 1:max_iter
            r1 = rand(3, n_particles); r2 = rand(3, n_particles);
            V = w*V + c1*r1.*(pbest - X) + c2*r2.*(gbest - X);
            X = X + V;
            X = max(min(X, ub'), lb');
            obj = arrayfun(@(k) objective(X(:,k), Time, Current, Voltage, Pocv, Ts, Q), 1:n_particles);
            mask = obj < pbest_obj;
            pbest(:, mask) = X(:, mask);
            pbest_obj(mask) = obj(mask);
            [new_best, idx] = min(pbest_obj);
            if new_best < gbest_obj
                gbest_obj = new_best;
                gbest = pbest(:, idx);
            end
        end
        results(i, j) = gbest_obj; % Store RMSE
        params_store(i, j, :) = gbest; % Store optimal parameters
        fprintf('lb_R0 = %.4f, lb_R1 = %.4f, RMSE = %.6f, R0 = %.5f, R1 = %.5f, C1 = %.2f\n', ...
            lb_R0, lb_R1, gbest_obj, gbest(1), gbest(2), gbest(3));
    end
end

% Simulate and overlay all predicted voltages
n = length(lb_values);
v_pred_all = cell(n, n); % Store predicted voltages
for i = 1:n
    for j = 1:n
        params = squeeze(params_store(i, j, :));
        [~, v_pred] = objective(params, Time, Current, Voltage, Pocv, Ts, Q);
        v_pred_all{i, j} = v_pred;
    end
end

figure;
plot(Time, Voltage, 'k', 'LineWidth', 2); hold on;
colors = lines(n*n); % Distinct colors for each prediction
count = 1;
for i = 1:n
    for j = 1:n
        plot(Time, v_pred_all{i, j}, '--', 'Color', colors(count,:), 'LineWidth', 1);
        count = count + 1;
    end
end
legend_entries = cell(1, n*n+1);
legend_entries{1} = 'Measured Voltage';
count = 2;
for i = 1:n
    for j = 1:n
        legend_entries{count} = sprintf('Predicted (lb_{R0}=%.4f, lb_{R1}=%.4f)', lb_values(i), lb_values(j));
        count = count + 1;
    end
end
legend(legend_entries, 'Location', 'best');
xlabel('Time [s]'); ylabel('Voltage [V]');
title('Overlay of Predicted Voltages for All Parameter Sets');
grid on;

% Heatmap for RMSE
figure;
imagesc(log10(results)); 
xticks(1:n); xticklabels(arrayfun(@num2str, lb_values, 'UniformOutput', false));
yticks(1:n); yticklabels(arrayfun(@num2str, lb_values, 'UniformOutput', false));
xlabel('lb\_R1'); ylabel('lb\_R0');
colorbar;
title('Log_{10}(RMSE) over lb\_R0 and lb\_R1');

% % Objective function
% function [J, v_sim] = objective(params, Time, Current, Voltage, Pocv, Ts, Q)
%     R0 = params(1); R1 = params(2); C1 = params(3);
%     z = 0.7;              % initial SOC
%     vRC = 0;              % RC branch voltage
%     v_sim = zeros(size(Time));
%     for k = 1:length(Time)
%         if k > 1
%             z = z + (Ts/Q) * Current(k-1);
%         end
%         vOC = polyval(Pocv, z);
%         alpha = exp(-Ts/(R1*C1));
%         vRC = alpha*vRC + R1*(1-alpha)*Current(k);
%         v_sim(k) = vOC + vRC + R0*Current(k);
%     end
%     J = mean((Voltage - v_sim).^2);
% end
