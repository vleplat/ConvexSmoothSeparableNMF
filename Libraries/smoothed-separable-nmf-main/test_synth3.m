% Test on synthetic data, p varies, mean vs median
clear all; clc; 

% Parameters
r = 10;
n = 1000; 
purity = 0.02; 
noislevel = 0.05; 
p = [1:20 20:10:160]; 
ntrials = 30; 

% Option to use mean instead of median
options.average = 1;

% Add utils and data 
addpath('./data'); 
addpath('./utils'); 

% load Wsynth; 
% [m,r] = size(W); 
% Wt = W;

load USGS_Library;
data = datalib(:,5:end); % first 4 cols of usgs are not materials
[m,nbcol] = size(data);

% Preallocate output arrays
err_svca = zeros(length(p), ntrials);
err_svca_mean = zeros(length(p), ntrials);
err_sspa = zeros(length(p), 1);
err_sspa_mean = zeros(length(p), 1);


% Perfom experiments
disp('*********************************')
for i = 1 : length(p)

    % Generate W
    selcol = randperm(nbcol, r);
    W = data(:,selcol);
    Wt = W;

    % Generate H and X
    Ht = [eye(r) sample_dirichlet(ones(r,1)*purity, n-r)'];
    X = Wt*Ht; 
    Noise = randn(m,n);  
    Xn = X + noislevel * Noise/norm(Noise,'fro') * norm(X,'fro');

    % Run SVCA
    for t = 1 : ntrials
        rng(t); % Seed for reproductible random
        % Run with median (default)
        [W,K] = SVCA(Xn, r, p(i));
        err_svca(i,t) = mrsaWs(Wt, W);
        % Run with mean
        [W,K] = SVCA(Xn, r, p(i), options);
        err_svca_mean(i,t) = mrsaWs(Wt, W);
    end

    % Run SSPA (only once because it is determinist)
    % Run with median (default)
    [W,K] = SSPA(Xn, r, p(i));
    err_sspa(i) = mrsaWs(Wt, W);
    % Run with mean
    [W,K] = SSPA(Xn, r, p(i), options);
    err_sspa_mean(i) = mrsaWs(Wt, W);
end

% Median of the error per parameter set over all trials
mederr_svca = median(err_svca, 2);
mederr_svca_mean = median(err_svca_mean, 2);

% Build matrix for convenient export to text (to use in latex/pgfplots)
results = [p' mederr_svca mederr_svca_mean err_sspa err_sspa_mean]

