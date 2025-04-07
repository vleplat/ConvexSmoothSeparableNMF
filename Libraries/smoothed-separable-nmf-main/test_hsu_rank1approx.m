% Test on Urban hsu, p varies, mean vs median vs rank-1 approx
clear all; clc; 

% Parameters
p = [1 200:200:6000]; 
ntrials = 10; 

% Option to use mean instead of median
options1.average = 1;
options3.average = 3;
options4.average = 4;


% Add utils and data 
addpath('./data'); 
addpath('./utils'); 

% Urban
data = load("./data/Urban.mat");
X = data.A; X = X';
r = 6;

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

    % Run SVCA
    for t = 1 : ntrials
        rng(t); % Seed for reproductible random
        % Run with median (default)
        [W,K] = SVCA(X, r, p(i));
        H = nnls_FPGM(X,W);
        err_svca(i,t) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
        % Run with mean
        [W,K] = SVCA(X, r, p(i), options1);
        H = nnls_FPGM(X,W);
        err_svca_mean(i,t) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
        % Run with t3
        [W,K] = SVCA(X, r, p(i), options3);
        H = nnls_FPGM(X,W);
        err_svca_t3(i,t) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
        % Run with t4
        [W,K] = SVCA(X, r, p(i), options4);
        H = nnls_FPGM(X,W);
        err_svca_t4(i,t) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
    end

    % Run SSPA (only once because it is determinist)
    % Run with median (default)
    [W,K] = SSPA(X, r, p(i));
    H = nnls_FPGM(X,W);
    err_sspa(i) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
    % Run with mean
    [W,K] = SSPA(X, r, p(i), options1);
    H = nnls_FPGM(X,W);
    err_sspa_mean(i) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
    % Run with t3
    [W,K] = SSPA(X, r, p(i), options3);
    H = nnls_FPGM(X,W);
    err_sspa_t3(i) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
    % Run with t4
    [W,K] = SSPA(X, r, p(i), options4);
    H = nnls_FPGM(X,W);
    err_sspa_t4(i) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
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













