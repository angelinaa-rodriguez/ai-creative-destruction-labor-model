%  LOAD & CLEAN DATA
data = readtable('final_merged_data.csv');

% Cutoff at April 2025
cutoff = datetime(2025,4,1);
data   = data(data.date <= cutoff, :);

% Remove rows with missing required variables
data = data(~isnan(data.employment) & ~isnan(data.wage_real) & ~isnan(data.value_added), :);

% Recompute time index
data.t = (0:height(data)-1)';

%% Extract variables
t     = data.t;
Edata = data.employment;
Wdata = data.wage_real;
Vdata = data.value_added / 1e6;   % scale to millions

%  INNOVATION A(t)
A0 = 2212.1;
g  = 0.008129;
pi_val = 0.017;


%  STARTING GUESSES (8 parameters)
start_alpha  = 2.0;
start_phi    = 0.35;
start_beta   = 0.0;
start_k      = 0.004;     % productivity scale
start_lambda = 0.009;      % saturation
start_delta  = 0.5;       % V/E effect
start_gamma  = 0.0002;    % E compression
start_rho    = 0.03;      % wage decay

par_guess = [start_alpha;
             start_phi;
             start_beta;
             start_k;
             start_lambda;
             start_delta;
             start_gamma;
             start_rho];

%  BOUNDS (8 parameters)
lb = [  0;    0;   -0.5;   0;     0;     0;      0;      0];
ub = [ 10;    2;    0.1;  1;   5;    50;    0.2;    0.2];

% Inequality constraint: k - lambda <= 0
Aineq = [0 0 0 1 -1 0 0 0];
bineq = 0;

options = optimoptions('fmincon', ...
    'Display','iter-detailed', ...
    'MaxFunctionEvaluations', 20000, ...
    'MaxIterations', 20000, ...
    'OptimalityTolerance', 1e-12, ...
    'StepTolerance', 1e-12);


%  RUN fmincon TO ESTIMATE PARAMETERS
[ParEst, ErrorVal] = fmincon( ...
    @(p) dist_EW(p, t, Edata, Wdata, Vdata, A0, g, pi_val), ...
    par_guess, ...
    Aineq, bineq, ...
    [], [], ...
    lb, ub, ...
    [], ...
    options);

disp('===== BEST FIT PARAMETERS =====');
disp(ParEst)
disp('===== FINAL SSE =====');
disp(ErrorVal);


%  SOLVE ODE WITH BEST PARAMETERS
Y0 = [Edata(1), Wdata(1)];

[tsol, Ysol] = ode45(@(tt,YY) econ_ode(tt,YY,ParEst,A0,g,Vdata,pi_val), ...
                     [t(1) t(end)], Y0);

E_model = interp1(tsol, Ysol(:,1), t);
W_model = interp1(tsol, Ysol(:,2), t);


%  FIT PLOTS
figure;
subplot(2,1,1)
plot(data.date,Edata,'bo'); hold on;
plot(data.date,E_model,'b-','LineWidth',2);
title('Employment Fit')

subplot(2,1,2)
plot(data.date,Wdata,'r+'); hold on;
plot(data.date,W_model,'r-','LineWidth',2);
title('Wage Fit')
xlabel('Date')


%  10-YEAR FORECAST
years_forward = 10;
months_forward = years_forward * 12;

t_future = (t(end) : t(end) + months_forward)';
t_full   = [t; t_future(2:end)];

Y0_full = [Edata(1), Wdata(1)];

[tsol_full, Ysol_full] = ode45(@(tt,YY) econ_ode(tt,YY,ParEst,A0,g,Vdata,pi_val), ...
                               t_full, Y0_full);

E_full = Ysol_full(:,1);
W_full = Ysol_full(:,2);

E_forecast = E_full(length(t)+1:end);
W_forecast = W_full(length(t)+1:end);

future_dates = data.date(end) + calmonths(1:months_forward);

figure;

subplot(2,1,1)
plot(data.date, Edata, 'bo'); hold on;
plot(data.date, E_full(1:length(t)), 'b-', 'LineWidth', 2);
plot(future_dates, E_forecast, 'b--', 'LineWidth', 2);
ylabel('Employment (thousands)')
title('Employment: Data, Fit, and 10-Year Projection')
grid on;

subplot(2,1,2)
plot(data.date, Wdata, 'r+'); hold on;
plot(data.date, W_full(1:length(t)), 'r-', 'LineWidth', 2);
plot(future_dates, W_forecast, 'r--', 'LineWidth', 2);
ylabel('Real Wage ($)')
xlabel('Date')
title('Real Wage: Data, Fit, and 10-Year Projection')
grid on;


%% DISTANCE FUNCTION
function d = dist_EW(par, t, Edata, Wdata, Vdata, A0, g, pi_val)

    Y0 = [Edata(1), Wdata(1)];

    try
        [~, Ysol] = ode45(@(tt,YY) econ_ode(tt,YY,par,A0,g,Vdata,pi_val), t, Y0);
    catch
        d = 1e12;
        return;
    end

    Emod = Ysol(:,1);
    Wmod = Ysol(:,2);

    if any(isnan(Emod)) || any(isnan(Wmod))
        d = 1e12;
        return;
    end

    d = sum((Emod - Edata).^2 + (Wmod - Wdata).^2);

end


%% ODE FUNCTION (WITH SATURATION)
function Ydot = econ_ode(t, Y, par, A0, g, Vdata, pi_val)

    E = max(Y(1), 1); 
    W = Y(2);

    % innovation curve
    A = A0 * exp(g*t);

    % pick correct V for this t
    idx = min(max(floor(t)+1,1), length(Vdata));
    V   = Vdata(idx);

    % unpack parameters
    alpha  = par(1);
    phi    = par(2);
    beta   = par(3);
    k      = par(4);
    lambda = par(5);
    delta  = par(6);
    gamma  = par(7);
    rho    = par(8);

    L = 4400;

    % EMPLOYMENT ODE
    dE = -alpha*(E/A) - pi_val*E + phi*(L - E) + beta*W;

    % WAGE ODE (WITH SATURATION)
    dW = (k*A)/(1 + lambda*A) + delta*(V/E) - gamma*E - rho*W;

    Ydot = [dE; dW];
end
