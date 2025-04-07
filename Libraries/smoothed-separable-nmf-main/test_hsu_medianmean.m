% Test on HSU data Cuprite, p varies, mean vs median
clear all; clc; 

% Parameters
p = [1 200:200:6000]; 
ntrials = 10; 

% Option to use mean instead of median
options.average = 1;

% Add utils
addpath('./utils'); 

% Cuprite
% data = load("./data/Cuprite.mat");
% X = data.x;
% r = 12;

% Urban
% data = load("./data/Urban.mat");
% X = data.A; X = X';
% r = 6;

% Terrain
data = load("./data/Terrain.mat");
X = data.A; X = X';
r = 6;

% Sandiego
% data = load("./data/SanDiego.mat");
% X = data.A; X = X';
% r = 8;

% Preprocess outliers
ball = [];  
for i = 1 : size(X,1) 
    [a,b] = sort(X(i,:),'descend'); 
    ball = [ball, b(1:10)]; 
end
ball = unique(ball); 
X(:,ball) = 0;
fprintf("%d outliers dismissed\n", length(ball));

% Preallocate output arrays
err_svca = zeros(length(p), ntrials);
err_svca_mean = zeros(length(p), ntrials);
err_sspa = zeros(length(p), 1);
err_sspa_mean = zeros(length(p), 1);


% Perfom experiments
for i = 1 : length(p)
    fprintf("%d ", p(i))

    % Run SVCA
    for t = 1 : ntrials
        rng(t); % Seed for reproductible random
        % Run with median (default)
        [W,K] = SVCA(X, r, p(i));
        H = NNLS(W,X);
        err_svca(i,t) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
        % Run with mean
        [W,K] = SVCA(X, r, p(i), options);
        H = NNLS(W,X);
        err_svca_mean(i,t) = norm(X-W*H,'fro')/norm(X,'fro')*100; 
    end

    % Run SSPA (only once because it is determinist)
    % Run with median (default)
    [W,K] = SSPA(X, r, p(i));
    H = NNLS(W,X);
    err_sspa(i) = norm(X-W*H,'fro')/norm(X,'fro')*100;
    % Run with mean
    [W,K] = SSPA(X, r, p(i), options);
    H = NNLS(W,X);
    err_sspa_mean(i) = norm(X-W*H,'fro')/norm(X,'fro')*100;
end

% Median of the error per parameter set over all trials
mederr_svca = median(err_svca, 2);
mederr_svca_mean = median(err_svca_mean, 2);
%minerr_svca = min(err_svca, 2);
%minerr_svca_mean = min(err_svca_mean, 2);

% Build matrix for convenient export to text (to use in latex/pgfplots)
%results = [p' mederr_svca mederr_svca_mean err_sspa err_sspa_mean]
results = [p' mederr_svca mederr_svca_mean err_sspa err_sspa_mean]

