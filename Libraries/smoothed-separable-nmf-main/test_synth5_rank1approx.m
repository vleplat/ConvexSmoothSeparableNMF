% Test on synthetic data, p varies, mean vs median vs rank1 approx
clear all; clc; 

% Parameters
n = 1000; 
purity = 0.02; 
noislevel = 0.05; 
p = [1:20 20:10:160]; 
ntrials = 30; 

% Option to use mean instead of median
options1.average = 1;
options3.average = 3;
options4.average = 4;


% Add utils and data 
addpath('./data'); 
addpath('./utils'); 

load Wsynth; 
[m,r] = size(W); 
Wt = W;

% Preallocate output arrays
err_svca = zeros(length(p), ntrials);
err_svca_mean = zeros(length(p), ntrials);
err_svca_t3 = zeros(length(p), ntrials);
err_svca_t4 = zeros(length(p), ntrials);
err_sspa = zeros(length(p), 1);
err_sspa_mean = zeros(length(p), 1);
err_sspa_t3 = zeros(length(p), 1);
err_sspa_t4 = zeros(length(p), 1);


% Perfom experiments
disp('*********************************')
for i = 1 : length(p)
    fprintf("%d ", p(i))
    
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
        [W,K] = SVCA(Xn, r, p(i), options1);
        err_svca_mean(i,t) = mrsaWs(Wt, W);
        % Run with t3
        [W,K] = SVCA(Xn, r, p(i), options3);
        err_svca_t3(i,t) = mrsaWs(Wt, W);
        % Run with t4
        [W,K] = SVCA(Xn, r, p(i), options4);
        err_svca_t4(i,t) = mrsaWs(Wt, W);
    end

    % Run SSPA (only once because it is determinist)
    % Run with median (default)
    [W,K] = SSPA(Xn, r, p(i));
    err_sspa(i) = mrsaWs(Wt, W);
    % Run with mean
    [W,K] = SSPA(Xn, r, p(i), options1);
    err_sspa_mean(i) = mrsaWs(Wt, W);
    % Run with t3
    [W,K] = SSPA(Xn, r, p(i), options3);
    err_sspa_t3(i) = mrsaWs(Wt, W);
    % Run with t4
    [W,K] = SSPA(Xn, r, p(i), options4);
    err_sspa_t4(i) = mrsaWs(Wt, W);
end

% Median of the error per parameter set over all trials
mederr_svca = median(err_svca, 2);
mederr_svca_mean = median(err_svca_mean, 2);
mederr_svca_t3 = median(err_svca_t3, 2);
mederr_svca_t4 = median(err_svca_t4, 2);

% Build matrix for convenient export to text (to use in latex/pgfplots)
results = [p' mederr_svca mederr_svca_mean err_sspa err_sspa_mean]

% Plot
legendstrings = ["SVCAmed" "SVCAavg" "SVCAt3" "SVCAt4"...
                 "SSPAmed" "SSPAavg" "SSPAt3" "SSPAt4"];
figure; 
semilogy(p',mederr_svca','o--'); hold on;
semilogy(p',mederr_svca_mean','o--');
semilogy(p',mederr_svca_t3','o--');
semilogy(p',mederr_svca_t4','o--');

semilogy(p',err_sspa','-o'); 
semilogy(p',err_sspa_mean','-o'); 
semilogy(p',err_sspa_t3','-o'); 
semilogy(p',err_sspa_t4','-o'); 

legend(legendstrings,'Interpreter','latex');
xlabel('$p$','Interpreter','latex'); 
ylabel('MRSA','Interpreter','latex');
hold off;













