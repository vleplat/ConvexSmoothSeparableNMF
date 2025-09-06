addpath(genpath(pwd));
clc
clear
disp('----------------------------------------------------------------------------------------------------------')
disp("HSI test - Jasper Ridge data set - Paper Section 5 ")
disp("The computation of X can take approximatively 30 minutes on a recent laptop...")
disp("You can directly load the .mat in ./Results/HSI/JasperRidge and start playing with the code starting from line 56")
disp('----------------------------------------------------------------------------------------------------------')

%%-------------------------------------------------------------------------
%% Load the data sets
%%-------------------------------------------------------------------------
% From https://lesun.weebly.com/hyperspectral-data-set.html
% Load the Ground-truth (W,H)
load('end4.mat') ;
W_true = M;
H_true = A;
clear M A 

% Load the HSI
load('jasperRidge2_R198.mat') ;
M = Y;
clear Y;

% Normalisation of M
% M=M./sum(M);
M = M/max(max(M));  % this scaling gives the best results

% Number of cluster/the factorization rank r
r = 4;

%%-------------------------------------------------------------------------
%% Parameters definition for algorithms
%%-------------------------------------------------------------------------
% Solver for CSSNMF (our method)
options.delta=0.02;
options.type=1;        % Defines the type of spectral clustering algorithm  that should be used. 
options.modeltype=1;   % NNLS
options.agregation = 0;
                     % 0 - average 
                     % 1 - median 
options.clustering = 0;
                     % 0 - spectral clustering
                     % 1 - kmeans clustering

% SSPA (from Nasidic et al., 2023)
nplp = 5; % actual value computed later

%%-------------------------------------------------------------------------
%% Run the algorithms
%%-------------------------------------------------------------------------
% Our method
% [X_fgnsr, K_fgnsr_1] = fgnsr_alg1(M, r, 'maxiter', 200);
[X_fgnsr, K_fgnsr_1] = fgnsr_alg1(M, r, 'maxiter', 400, 'mu',80);

%% ------------------------------------------------------------------------
%% Help choosing a value for delta (CSSNMF)
%% ------------------------------------------------------------------------
close all
histogram(sum(X_fgnsr,2));

%% Run CSSNMF post-processing (Alg.2)
options.delta      = 1.15;     % e.g., 1.11 or 1.30 also work
options.agregation = 0;        % 0 average, 1 median
options.clustering = 0;        % 0 spectral, 1 k-means
[W_fgnsr, H_fgnsr, K_fgnsr, Wfgnsr] = alg2(M, X_fgnsr, r, options);

% Relative Frobenius error for CSSNMF
res_fgnsr = norm(M - W_fgnsr*H_fgnsr, 'fro') / norm(M, 'fro');

%% ------------------------------------------------------------------------
%% SPA (anchors) + NNLS
%% ------------------------------------------------------------------------
K_spa_anch = FastSepNMF(M, r, 1);           % anchor indices
W_spa_anch = M(:, K_spa_anch);              % endmembers from anchors
H_spa_anch = nnlsHALSupdt_new(W_spa_anch' * M, W_spa_anch, [], 1000);

res_spa_anch = norm(M - W_spa_anch*H_spa_anch, 'fro') / norm(M, 'fro');

%% ------------------------------------------------------------------------
%% SSPA (grid search for best nplp) + NNLS
%% ------------------------------------------------------------------------
res_sspa_list = [];
nplp_grid     = 10:50:1200;                 % coarse grid for Jasper Ridge
for nplp = nplp_grid
    [W_tmp, ~] = SSPA(M, r, nplp);
    H_tmp      = nnlsHALSupdt_new(W_tmp' * M, W_tmp, [], 1000);
    res_sspa_list(end+1) = norm(M - W_tmp*H_tmp, 'fro') / norm(M, 'fro'); %#ok<SAGROW>
end

% Display error vs nplp and pick the best
figure; plot(nplp_grid, res_sspa_list, '-o'); grid on
xlabel('nplp'); ylabel('Rel. Frobenius Error');
[~, best_idx] = min(res_sspa_list);
nplp_best = nplp_grid(best_idx);            % e.g., 910 from your note

% Final SSPA at best nplp
Options.average = 0; % 1 = mean, 0 = median (default/robust)
[W_sspa, K_sspa] = SSPA(M, r, nplp_best, Options);
H_sspa           = nnlsHALSupdt_new(W_sspa' * M, W_sspa, [], 1000);
res_sspa         = norm(M - W_sspa*H_sspa, 'fro') / norm(M, 'fro');

%% ------------------------------------------------------------------------
%% Alignments to ground truth (W,H) and metrics (SSIM, d_W)
%% ------------------------------------------------------------------------
% Align abundances to A (H_true) and endmembers to W_true
A = H_true;
B = W_true;

% CSSNMF
H_fgnsr_re = matchCol(H_fgnsr', A')';
W_fgnsr_re = matchCol(W_fgnsr, B);

% SPA
H_spa_anch_re = matchCol(H_spa_anch', A')';
W_spa_anch_re = matchCol(W_spa_anch, B);

% SSPA
H_sspa_re = matchCol(H_sspa', A')';
W_sspa_re = matchCol(W_sspa, B);

% Normalize endmembers columnwise (l1) for d_W
normalize_cols = @(Z) Z ./ sum(Z,1);
gt  = normalize_cols(B);            % ground-truth normalized
y_c = normalize_cols(W_fgnsr_re);   % CSSNMF
y_a = normalize_cols(W_spa_anch_re);% SPA
y_s = normalize_cols(W_sspa_re);    % SSPA

% Endmember relative error d_W (in percent for printing)
dW_cssnmf = 100 * norm(y_c - gt, 'fro') / norm(gt, 'fro');
dW_spa    = 100 * norm(y_a - gt, 'fro') / norm(gt, 'fro');
dW_sspa   = 100 * norm(y_s - gt, 'fro') / norm(gt, 'fro');

% SSIM (global) on abundance maps (already aligned)
ssim_CSSNMF = ssim(H_fgnsr_re, A);
ssim_SPA    = ssim(H_spa_anch_re, A);
ssim_SSPA   = ssim(H_sspa_re, A);

%% ------------------------------------------------------------------------
%% Display some results: abundance maps
%% ------------------------------------------------------------------------
close all
% Each call shows the r maps of a method in a grid. Keep four separate figures.
affichage(H_true',     r, 100, 100); title('Ground truth (abundances)');
affichage(H_fgnsr_re', r, 100, 100); title('CSSNMF (abundances)');
affichage(H_spa_anch_re', r, 100, 100); title('SPA (abundances)');
affichage(H_sspa_re',  r, 100, 100); title('SSPA (abundances)');

%% ------------------------------------------------------------------------
%% Display spectral signatures (2x2 grid: GT / CSSNMF / SPA / SSPA)
%% ------------------------------------------------------------------------
close all
x         = 1:size(M,1);
colors    = {'b','k','r','g','c','m'};
linestyle = {'-','--','-.',':'};
getFirst  = @(v)v{1};
getprop   = @(v,idx)getFirst(circshift(v, -idx+1));
linew     = 1.5; fontSize = 14;

figure
% (1) GT
subplot(2,2,1)
hold on
for t = 1:r
    plot(x, gt(:,t), 'color', getprop(colors,t), 'linestyle', getprop(linestyle,t), 'LineWidth', linew);
end
set(gca,'xlim',[1 numel(x)]); grid on
title('Ground truth','Interpreter','latex','FontSize',fontSize);
legend('1','2','3','4','Interpreter','latex','FontSize',fontSize,'Location','best'); % adjust labels for materials

% (2) CSSNMF
subplot(2,2,2)
hold on
for t = 1:r
    plot(x, y_c(:,t), 'color', getprop(colors,t), 'linestyle', getprop(linestyle,t), 'LineWidth', linew);
end
set(gca,'xlim',[1 numel(x)]); grid on
title('CSSNMF','Interpreter','latex','FontSize',fontSize);

% (3) SPA
subplot(2,2,3)
hold on
for t = 1:r
    plot(x, y_a(:,t), 'color', getprop(colors,t), 'linestyle', getprop(linestyle,t), 'LineWidth', linew);
end
set(gca,'xlim',[1 numel(x)]); grid on
title('SPA','Interpreter','latex','FontSize',fontSize);
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize);

% (4) SSPA
subplot(2,2,4)
hold on
for t = 1:r
    plot(x, y_s(:,t), 'color', getprop(colors,t), 'linestyle', getprop(linestyle,t), 'LineWidth', linew);
end
set(gca,'xlim',[1 numel(x)]); grid on
title(sprintf('SSPA (nplp = %d)', nplp_best),'Interpreter','latex','FontSize',fontSize);
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize);

%% ------------------------------------------------------------------------
%% Console summary
%% ------------------------------------------------------------------------
clc
disp('----------------------------- CSSNMF |    SPA    |   SSPA   -----------------------------')
fprintf('Rel. Frob Error (lower is better):  %8.4e | %8.4e | %8.4e\n', res_fgnsr, res_spa_anch, res_sspa);
fprintf('SSIM (higher is better):            %8.2f | %8.2f | %8.2f\n', 100*ssim_CSSNMF, 100*ssim_SPA, 100*ssim_SSPA);
fprintf('||W - W#||_F / ||W#||_F (%%):        %8.2f | %8.2f | %8.2f\n', dW_cssnmf, dW_spa, dW_sspa);
disp('-----------------------------------------------------------------------------------------')


