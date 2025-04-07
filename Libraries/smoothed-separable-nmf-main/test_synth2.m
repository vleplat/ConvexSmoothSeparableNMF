% NOT USED ANYMORE
% Test on synthetic data, mean vs median 
clear all; clc; 

% Parameters
n = 1000; 
purity = 0.05; 
noislevel = logspace(-2,0,20); 
p = [20 50]; 
ntrials = 30; 

% Add utils and data 
addpath('./data'); 
addpath('./utils'); 

load Wsynth; 
[m,r] = size(W); 
Wt = W;

% Preallocate output arrays
err_svca_med = zeros(length(p), length(noislevel), ntrials);
err_svca_avg = zeros(length(p), length(noislevel), ntrials);
err_sspa_med = zeros(length(p), length(noislevel));
err_sspa_avg = zeros(length(p), length(noislevel));

% Option to use mean instead of median
options.average = 1;

% Perfom experiments
disp('*********************************')
for i = 1 : length(noislevel)
    
    % Generate H and X
    Ht = [eye(r) sample_dirichlet(ones(r,1)*purity, n-r)'];
    X = Wt*Ht; 
    Noise = randn(m,n);  
    Xn = X + noislevel(i) * Noise/norm(Noise,'fro') * norm(X,'fro');
    
    for j = 1 : length(p)
        % Run SVCA
        for t = 1 : ntrials
            rng(t); % Seed for reproductible random
            % Run with median (default)
            [W,K] = SVCA(Xn, r, p(j));
            err_svca_med(j,i,t) = mrsaWs(Wt, W);
            % Run with mean/average
            [W,K] = SVCA(Xn, r, p(j), options);
            err_svca_avg(j,i,t) = mrsaWs(Wt, W);
        end
        
        % Run SSPA (only once because it is determinist)
        % Run with median (default)
        [W,K] = SSPA(Xn, r, p(j));
        err_sspa_med(j,i) = mrsaWs(Wt, W);
        % Run with mean/average
        [W,K] = SSPA(Xn, r, p(j), options);
        err_sspa_avg(j,i) = mrsaWs(Wt, W);
    end

end

% Median of the error per parameter set over all trials
mederr_svca_med = median(err_svca_med, 3);
mederr_svca_avg = median(err_svca_avg, 3);

% Build matrix for convenient export to text (to use in latex/pgfplots)
dataout = [noislevel' mederr_svca_med' mederr_svca_avg' err_sspa_med' err_sspa_avg']

