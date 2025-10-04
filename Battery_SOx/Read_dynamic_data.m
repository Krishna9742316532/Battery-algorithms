function [D_FUDS, D_HDS, D_BJDST] = Read_dynamic_data ()
% Read original data
% [num,txt] = xlsread('2FUDS_25C_80SOC.xls', 'Channel_1-008');
% [num1,txt1] = xlsread('3US06_HDS_25C_80SOC.xls', 'Channel_1-008');
% [num2,txt2] = xlsread('4BJDST_25C_80SOC.xls', 'Channel_1-008');

% Use the readed Data from above step, load them here and do not need read original data every time.
  load ('2FUDS_25C_80SOC.mat')
  load ('3US06_HDS_25C_80SOC.mat')
  load ('4BJDST_25C_80SOC.mat')

%Get the 'time(s)', 'voltage(V)' and 'current(A)' for three dynamic tests from the data load in above step.
% 2-FUDS (Federal Urban Driving Schedule)
  t = num(:,2);     % Test time (s)
  I = num(:,7);     % Current (A), charge is plus and discharge is minus
  V = num(:,8);     % Terminal Voltage (V)
% Choose start number when the dynamic tests begin and then the SOC is 80% for the Datasets.
  ST = 2584;           % Start number, for 25C.
  I2 = I(ST:end);      % (A)
  V2 = V(ST:end);   % (V)
  t2 = t(ST:end);      % (s)
  t2 = t2 - t2(1);       % Make the time start from 0.

% 3-HDS (Highway Driving Schedule)
  t = num1(:,2);    % Test time (s)
  I = num1(:,7);    % Current (A), charge is plus and discharge is minus
  V = num1(:,8);    % Terminal Voltage (V)
% Choose start number
  ST = 1210;        % Start number, for 25C.
  I3 = I(ST:end);   % (A)
  V3 = V(ST:end);   % (V)
  t3 = t(ST:end);   % (s)
  t3 = t3 - t3(1);  % Make the time start from 0.

% 4BJDST
  t = num2(:,2);    % Test time (s)
  I = num2(:,7);    % Current (A), charge is plus and discharge is minus
  V = num2(:,8);    % Terminal Voltage (V)

% Choose start number
  ST = 1224;        % Start number, for 25C.
  I4 = I(ST:end);   % (A) 
  V4 = V(ST:end);   % (V)
  t4 = t(ST:end);   % (s)
  t4 = t4 - t4(1);  % Make the time start from 0.

% Summary all the Data of the three dynamic tests
  D_FUDS = [t2, I2, V2];   % Data for Federal Urban Driving Schedule
  D_HDS = [t3, I3, V3];    % Data for Highway Driving Schedule
  D_BJDST = [t4, I4, V4];  % Data for Beijing Dynamic Stress Test
  
end