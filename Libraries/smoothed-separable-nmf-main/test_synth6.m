% Test on synthetic data alls and svca worst/med/best
clear all; clc; 

% Parameters
n = 1000; 
purity = 0.05; 
noislevel = logspace(-2,0,20); 
p = 30; 
ntrials = 30;

% Add utils and data 
addpath('./data'); 
addpath('./utils'); 

load Wsynth; 
[m,r] = size(W); 
Wt = W;

% Preallocate output arrays
err_alls = zeros(length(noislevel), ntrials);
err_svca = zeros(length(noislevel), ntrials);
recerr_svca = zeros(length(noislevel), ntrials);
err_sspa = zeros(length(noislevel), 1);


% Perfom experiments
disp('*********************************')
for i = 1 : length(noislevel)

            % Generate H and X
            Ht = [eye(r) sample_dirichlet(ones(r,1)*purity, n-r)'];
            X = Wt*Ht; 
            Noise = randn(m,n);  
            Xn = X + noislevel(i) * Noise/norm(Noise,'fro') * norm(X,'fro');

            % Run ALLS
            for t = 1 : ntrials
                rng(t); % Seed for reproductible random
                [W,K] = ALLS(Xn, r, p);
                err_alls(i,t) = mrsaWs(Wt, W);
            end

            % Run SVCA
            for t = 1 : ntrials
                rng(t); % Seed for reproductible random
                [W,K] = SVCA(Xn, r, p);
                err_svca(i,t) = mrsaWs(Wt, W);
                H = NNLS(W,Xn);
                recerr_svca(i,t) = norm(X-W*H,'fro')/norm(X,'fro')*100;
            end

            % Run SSPA (only once because it is determinist)
            [W,K] = SSPA(Xn, r, p);
            err_sspa(i) = mrsaWs(Wt, W);
end

% Median of the error per parameter set over all trials
worsterr_alls = max(err_alls, [], 2);
mederr_alls = median(err_alls, 2);
besterr_alls = min(err_alls, [], 2);

worsterr_svca = max(err_svca, [], 2);
mederr_svca = median(err_svca, 2);
besterr_svca = min(err_svca, [], 2);

% MRSA corresponding to the solutions with best relative reconstr error
bestrecerr_svca = zeros(length(noislevel), 1);
for i = 1:length(noislevel)
    [~,idx] = min(recerr_svca(i,:));
    bestrecerr_svca(i) = err_svca(i,idx);
end

% Build matrix for convenient export to text (to use in latex/pgfplots)
results = [noislevel' worsterr_alls mederr_alls besterr_alls...
           worsterr_svca mederr_svca besterr_svca err_sspa bestrecerr_svca]




