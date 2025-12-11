clear; close all; clc;

%% grab data 

data = readtable('final_merged_data.csv');

cutoff = datetime(2025,4,1);
data   = data(data.date <= cutoff, :);

% Remove rows with missing required variables
data = data(~isnan(data.employment) & ~isnan(data.wage_real) & ~isnan(data.value_added), :);

% Recompute time index
data.t = (0:height(data)-1)';

% Extract variables
t     = data.t;
Edata = data.employment;
Wdata = data.wage_real;
Vdata = data.value_added / 1e6;  


%% grab the fitted parameters

params0.phi    = 0.0539;
params0.beta   = -0.0053;  
params0.delta  = 10.0000;  
params0.gamma  = 0.0000;
params0.rho    = 0.0000;


% grab other inputs used by the model
params0.Vdata = Vdata;  
params0.E0 = Edata(1);
params0.W0 = Wdata(1);
params0.tgrid = t;     
params0.A0 = 0;
params0.g  = 0;

L = 4400;

%% set estimation bounds for variables (for LHS sampling)
% Order: phi, beta, delta, gamma, rho
% Choose sensible bounds for each parameter (edit to your priors)
est_lb =  [0; -0.1; 0; 0; 0];   % lower bounds
est_ub = [2; 0.1; 50; 1; 1];   % upper bounds, increased from model code to do meaningful sensitivity analysis

% parameter names in same order for sampling
sampleParams = {'phi','beta','delta','gamma','rho'};


%% local sensitivity -> elasticity
T = t(end);         
eps_frac = 1e-4;    

baseP = params0;
[E_base_final, W_base_final] = simulate_model(baseP, L, T);

local_results = struct();
fnames = fieldnames(params0);
for k = 1:numel(fnames)
    name = fnames{k};
    if ismember(name, {'Vdata','E0','W0','tgrid','A0','g'})
        continue;
    end
    baseval = params0.(name);
    dp = eps_frac * max(abs(baseval), 1e-8);
    p_plus = params0; p_minus = params0;
    p_plus.(name)  = baseval + dp;
    p_minus.(name) = baseval - dp;
    [E_p, W_p] = simulate_model(p_plus, L, T);
    [E_m, W_m] = simulate_model(p_minus, L, T);
    dEdp = (E_p - E_m) / (2*dp);
    dWdp = (W_p - W_m) / (2*dp);
    elasticity_E = (baseval / E_base_final) * dEdp;
    elasticity_W = (baseval / W_base_final) * dWdp;
    local_results.(name) = struct('dEdp',dEdp,'dWdp',dWdp,...
                                 'elasticity_E',elasticity_E,'elasticity_W',elasticity_W);
end


names = fieldnames(local_results);

% print results to grab exact values
fprintf('\nLOCAL RESULTS\n');
for k = 1:numel(names)
    nm = names{k};
    lr = local_results.(nm);
    fprintf('%s:  dE/dp = %.4f,  dW/dp = %.4f,  elast_E = %.4f,  elast_W = %.4f\n', ...
        nm, lr.dEdp, lr.dWdp, lr.elasticity_E, lr.elasticity_W);
end

% plot local E_final
fprintf('\n');
elsE = cellfun(@(n) local_results.(n).elasticity_E, names);
[elsE_sorted, idxE] = sort(elsE);
figure; barh(elsE_sorted); set(gca,'YTick',1:numel(names),'YTickLabel',names(idxE));
xlabel('Elasticity on E(T)'); title('Local sensitivity for E(T)'); grid on;

% plot local W_final
elsW = cellfun(@(n) local_results.(n).elasticity_W, names);
[elsW_sorted, idxW] = sort(elsW);
figure; barh(elsW_sorted); set(gca,'YTick',1:numel(names),'YTickLabel',names(idxW));
xlabel('Elasticity on W(T)'); title('Local sensitivity for W(T)'); grid on;

%% global sensitivity -> lhs + PRCC 

N = 1000;           
frac = 0.20;        

M = numel(sampleParams);
lhsU = lhsdesign(N, M, 'criterion','maximin','iterations',1000);

% bounds for sampling 
% bounds for sampling 
lb = zeros(1,M); ub = zeros(1,M);
for i=1:M
    pname = sampleParams{i};
    v = params0.(pname);
    
    if abs(v) < 1e-12
        % use estimation bounds for near-zero parameters
        switch pname
            case 'phi'; idx_est = 1;
            case 'beta';   idx_est = 2;
            case 'delta';    idx_est = 3;
            case 'gamma';  idx_est = 4;
            case 'rho';    idx_est = 5;
        end
        lb(i) = est_lb(idx_est);
        ub(i) = est_ub(idx_est);
    else
        % use fraction around baseline for non-zero parameters
        lb(i) = v * (1 - frac);
        ub(i) = v * (1 + frac);
    end
    
    % ensure lb < ub
    if lb(i) >= ub(i)
        mid = (lb(i) + ub(i))/2;
        lb(i) = mid - 0.01;
        ub(i) = mid + 0.01;
    end
end

paramSamples = bsxfun(@plus, lb, bsxfun(@times, lhsU, (ub-lb)));

% run model for each sample
E_out = nan(N,1); W_out = nan(N,1);
for i=1:N
    p_i = params0;
    for j=1:M
        p_i.(sampleParams{j}) = paramSamples(i,j);
    end
    try
        [E_out(i), W_out(i)] = simulate_model(p_i, L, T);
    catch ME
        warning('ODE failed at sample %d: %s', i, ME.message);
        E_out(i) = NaN; W_out(i) = NaN;
    end
end

% get rid of non valid runs
valid = ~isnan(E_out) & ~isnan(W_out);
paramSamples_valid = paramSamples(valid,:);
E_out = E_out(valid);
W_out = W_out(valid);

% remove parameter columns with basically zero variance
varMask = var(paramSamples_valid,1) > 1e-12;  
paramSamples_clean = paramSamples_valid(:, varMask);
paramsUsed = sampleParams(varMask);

    
% rank stuff 
Xrank = tiedrank(paramSamples_clean);
yrankE = tiedrank(E_out);
yrankW = tiedrank(W_out);

% PRCC for E_final
MmatE = [Xrank, yrankE];
R = corrcoef(MmatE);
Rinv = pinv(R);
nparams = size(Xrank,2);
prccE = zeros(nparams,1);
for i=1:nparams
    prccE(i) = -Rinv(i,end) / sqrt(Rinv(i,i) * Rinv(end,end));
end

% PRCC for W_final
MmatW = [Xrank, yrankW];
R2 = corrcoef(MmatW);
Rinv2 = pinv(R2);
prccW = zeros(nparams,1);
for i=1:nparams
    prccW(i) = -Rinv2(i,end) / sqrt(Rinv2(i,i) * Rinv2(end,end));
end

% print results to grab exact values
fprintf('\nGLOBAL RESULTS\n');
for i = 1:numel(paramsUsed)
    fprintf('%s:  PRCC_E = %.4f,  PRCC_W = %.4f\n', ...
        paramsUsed{i}, prccE(i), prccW(i));
end
fprintf('\n');

% plot PRCC for E
[sortedE, idxE] = sort(prccE);
figure; barh(sortedE);
set(gca,'YTick',1:sum(varMask),'YTickLabel',paramsUsed(idxE));
xlabel('PRCC'); title('Global Sensitivity for E(T)'); grid on;

% plot PRCC for W
[sortedW, idxW] = sort(prccW);
figure; barh(sortedW);
set(gca,'YTick',1:sum(varMask),'YTickLabel',paramsUsed(idxW));
xlabel('PRCC'); title('Global Sensitivity for W(T)'); grid on;

%% results yum
out.paramSamples = array2table(paramSamples_clean, 'VariableNames', paramsUsed);
out.E_out = E_out; out.W_out = W_out;
out.prccE = array2table(prccE', 'VariableNames', paramsUsed);
out.prccW = array2table(prccW', 'VariableNames', paramsUsed);
out.local = local_results;
save('sensitivity_results_newmodel.mat','out');

writetable(out.paramSamples, 'param_samples_newmodel.csv');
writematrix([E_out, W_out], 'outputs_newmodel.csv');
TBL = table(paramsUsed', prccE, prccW, 'VariableNames',{'Parameter','PRCC_E','PRCC_W'});
writetable(TBL,'prcc_summary_newmodel.csv');

fprintf('successful run yay.\n');

%% simulation 

function [Efinal, Wfinal] = simulate_model(p, L, T)
    y0 = [p.E0, p.W0];
    opts = odeset('RelTol',1e-6,'AbsTol',1e-8);

    [tsol, Y] = ode45(@(tt,YY) econ_ode(tt,YY,p,L), [0 T], y0, opts);

    Efinal = Y(end,1);
    Wfinal = Y(end,2);
end

%% updated ode function for the new model
function Ydot = econ_ode(t, Y, p, L)

    E = Y(1);
    W = Y(2);

    % interpolate V just like pre-AI model
    V = interp1(p.tgrid, p.Vdata, t, 'linear', 'extrap');

    % parameters 
    phi   = p.phi;
    beta  = p.beta;
    delta = p.delta;
    gamma = p.gamma;
    rho   = p.rho;

    % prevent V/E from becoming too big
    if E < 1e-6
        E = 1e-6;
    end

    dE = phi * (L - E) - 0.017 * E + beta * W;
    dW = delta * (V/E) - gamma * E - rho * W;

    Ydot = [dE; dW];
end


