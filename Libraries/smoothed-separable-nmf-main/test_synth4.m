% Test on synthetic data, p varies, different purities
clear all; clc; 

% Parameters
n = 1000; 
purities = [0.01 0.02 0.05]; 
noislevel = 0.05; 
p = [1:20 20:10:160]; 
ntrials = 30;

% Add utils and data 
addpath('./data'); 
addpath('./utils'); 

load Wsynth; 
[m,r] = size(W); 
Wt = W;

% Preallocate output arrays
err_svca = zeros(length(purities), length(p), ntrials);
err_sspa = zeros(length(purities), length(p), 1);


% Perfom experiments
disp('*********************************')
for i = 1 : length(purities)
    fprintf('purity %d', purities(i))
    for j = 1 : length(p)
        % Generate H and X
        Ht = [eye(r) sample_dirichlet(ones(r,1)*purities(i), n-r)'];
        X = Wt*Ht;
        Noise = randn(m,n);
        Xn = X + noislevel * Noise/norm(Noise,'fro') * norm(X,'fro');
        
        % Run SVCA
        for t = 1 : ntrials
            rng(t); % Seed for reproductible random
            % Run with median (default)
            [W,K] = SVCA(Xn, r, p(j));
            err_svca(i,j,t) = mrsaWs(Wt, W);
        end
        
        % Run SSPA (only once because it is determinist)
        % Run with median (default)
        [W,K] = SSPA(Xn, r, p(j));
        err_sspa(i,j) = mrsaWs(Wt, W);
    end
end

% Median of the error per parameter set over all trials
mederr_svca = median(err_svca, 3);

% Build matrix for convenient export to text (to use in latex/pgfplots)
results = [p' mederr_svca' err_sspa']

