addpath(genpath(pwd));
clc
clear
disp('----------------------------------------------------------------------------------------------------------')
disp("HSI test - Urban data set - Paper Section 5.2 ")
disp("The computation of X can take approximatively 45 minutes on a recent laptop...")
disp("You can directly load the .mat ./Results/HSI/Urban and start playing with the code starting from line 59")
disp('----------------------------------------------------------------------------------------------------------')

%%-------------------------------------------------------------------------
%% Load the data sets
%%-------------------------------------------------------------------------
% Load the Ground-truth (W,H)
load('end6_groundTruth.mat') ;
W_true = M;
H_true = A;
clear M A B

% Load the HSI
load('Urban.mat') ;
M = A';
clear A;

% Normalisation of M
M=M./sum(M);

% Number of cluster/the factorization rank r
r = 6;

% Perform Subsampling -> build Ms = M(:,1:HopSize:n);
[m,n] = size(M);
HopSize = 9;
Ms = M(:,1:HopSize:n);

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
nplp = 150; % actual value computed later

%%-------------------------------------------------------------------------
%% Run the algorithms
%%-------------------------------------------------------------------------
% Our method
% [X_fgnsr, K_fgnsr_1] = fgnsr_alg1(M, r, 'maxiter', 200);
[X_fgnsr, K_fgnsr_1] = fgnsr_alg1(Ms, r, 'maxiter', 400, 'mu',0.5);

%% ------------------------------------------------------------------------
%% Urban (full image) – CSSNMF vs SPA vs SSPA (tuned nplp)
%% Assumes M (162 x 307*307), W_true (162 x 6), H_true (6 x 307*307), r = 6
%% ------------------------------------------------------------------------
% ------------ CSSNMF (Alg. 1 + Alg. 2 postproc) --------------------------
% Compute X (already done earlier typically). Example:
% [X_fgnsr, ~] = fgnsr_alg1(M, r, 'maxiter', 400, 'mu', 80);

% Help choosing delta
close all
histogram(sum(X_fgnsr,2)); grid on; title('Histogram of row-sums of X');

% Final delta (set after quick visual scan or micro-grid)
options.delta      = 1.05;   % was ~1.04–1.10 in earlier runs; adjust if needed
options.type       = 1;      % spectral clustering variant
options.modeltype  = 1;      % NNLS (mixed model)
options.agregation = 0;      % 0 average, 1 median  (Urban often favored average)
options.clustering = 0;      % 0 spectral, 1 kmeans

[W_css,H_css_s,~,~] = alg2(Ms, X_fgnsr, r, options);
H_css=nnlsHALSupdt_new(W_css'*M,W_css,[],1000);
res_css = norm(M - W_css*H_css, 'fro') / norm(M, 'fro');
%% If data are loaded from .mat available online
res_css = res_fgnsr;
W_css = W_fgnsr;
H_css = H_fgnsr;
%%
% ------------ SPA baseline ------------------------------------------------
K_spa  = FastSepNMF(Ms, r, 0);
W_spa  = Ms(:, K_spa);
H_spa  = nnlsHALSupdt_new(W_spa' * M, W_spa, [], 1000);
res_spa = norm(M - W_spa*H_spa, 'fro') / norm(M, 'fro');

% ------------ SSPA (grid over nplp, then final run) ----------------------
res_sspa_grid = [];
nplp_grid = 10:50:1000;      % adjust grid if runtime is too long
for nplp = nplp_grid
    [W_tmp, ~]  = SSPA(Ms, r, nplp);
    H_tmp       = nnlsHALSupdt_new(W_tmp' * M, W_tmp, [], 1000);
    res_sspa_grid(end+1) = norm(M - W_tmp*H_tmp, 'fro') / norm(M, 'fro'); 
end

figure; plot(nplp_grid, res_sspa_grid, '-o'); grid on
xlabel('nplp'); ylabel('Rel. Frob. Error');
title('SSPA: Rel. Frob. Error vs nplp (Urban, full image)');

[~, idx_best] = min(res_sspa_grid);
nplp_best = nplp_grid(idx_best);

Options.average = 0; % 0 median (default), 1 mean — pick per map quality if needed
[W_sspa, ~] = SSPA(M, r, nplp_best, Options);
H_sspa      = nnlsHALSupdt_new(W_sspa' * M, W_sspa, [], 1000);
res_sspa    = norm(M - W_sspa*H_sspa, 'fro') / norm(M, 'fro');

%%
% ------------ Align to ground truth & build displays ---------------------
A = H_true;
% Reorder H-rows to match GT (and W-columns likewise)
H_css_re  = matchCol(H_css',  A')';
H_spa_re  = matchCol(H_spa',  A')';
H_sspa_re = matchCol(H_sspa', A')';

W_css_re  = matchCol(W_fgnsr,   W_true);
W_spa_re  = matchCol(W_spa,   W_true);
W_sspa_re = matchCol(W_sspa,  W_true);

% ------------ Abundance maps (scale each row by its max for visualization)
close all
affichage((diag(1./max(H_true,   [],2)) * H_true   )', r, 307, 307);  % GT
affichage((diag(1./max(H_css_re, [],2)) * H_css_re )', r, 307, 307);  % CSSNMF
affichage((diag(1./max(H_spa_re, [],2)) * H_spa_re )', r, 307, 307);  % SPA
affichage((diag(1./max(H_sspa_re,[],2)) * H_sspa_re)', r, 307, 307);  % SSPA

% ------------ Spectral signatures (2x2 tiled figure)
close all
x = 1:162;
colors    = {'b','k','r','g','c','m'};
linestyle = {'-','--','-.',':','-','-.'};
getFirst = @(v)v{1}; 
getprop  = @(opt, idx)getFirst(circshift(opt,-idx+1));
linew = 1.5; fontSize = 14;

tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% (1) Ground truth
nexttile;
for t=1:r
    gt(:,t) = W_true(:,t) / sum(W_true(:,t)); hold on
    plot(x, gt(:,t), 'Color', getprop(colors,t), 'LineStyle', getprop(linestyle,t), 'LineWidth', linew);
    set(gca,'xlim',[1 162]);
end
title('Ground truth','Interpreter','latex','FontSize',fontSize);
legend('Asphalt road','Grass','Tree','Roof','Metal','Dirt','Interpreter','latex','FontSize',fontSize);
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize); grid on

% (2) CSSNMF
nexttile;
for t=1:r
    y_css(:,t) = W_css_re(:,t) / sum(W_css_re(:,t)); hold on
    plot(x, y_css(:,t), 'Color', getprop(colors,t), 'LineStyle', getprop(linestyle,t), 'LineWidth', linew);
    set(gca,'xlim',[1 162]);
end
title('CSSNMF','Interpreter','latex','FontSize',fontSize);
legend('Asphalt road','Grass','Tree','Roof','Metal','Dirt','Interpreter','latex','FontSize',fontSize);
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize); grid on

% (3) SPA
nexttile;
for t=1:r
    y_spa(:,t) = W_spa_re(:,t) / sum(W_spa_re(:,t)); hold on
    plot(x, y_spa(:,t), 'Color', getprop(colors,t), 'LineStyle', getprop(linestyle,t), 'LineWidth', linew);
    set(gca,'xlim',[1 162]);
end
title('SPA','Interpreter','latex','FontSize',fontSize);
legend('Asphalt road','Grass','Tree','Roof','Metal','Dirt','Interpreter','latex','FontSize',fontSize);
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize); grid on

% (4) SSPA
nexttile;
for t=1:r
    y_sspa(:,t) = W_sspa_re(:,t) / sum(W_sspa_re(:,t)); hold on
    plot(x, y_sspa(:,t), 'Color', getprop(colors,t), 'LineStyle', getprop(linestyle,t), 'LineWidth', linew);
    set(gca,'xlim',[1 162]);
end
title(sprintf('SSPA (nplp = %d)', nplp_best),'Interpreter','latex','FontSize',fontSize);
legend('Asphalt road','Grass','Tree','Roof','Metal','Dirt','Interpreter','latex','FontSize',fontSize);
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize); grid on

% ------------ SSIM (global) ---------------------------------------------
ssim_css  = ssim(H_css_re,  A);
ssim_spa  = ssim(H_spa_re,  A);
ssim_sspa = ssim(H_sspa_re, A);

% ------------ Distances to W ground truth (percent) ----------------------
dW_css  = norm(y_css  - gt, 'fro') / norm(gt,'fro') * 100;
dW_spa  = norm(y_spa  - gt, 'fro') / norm(gt,'fro') * 100;
dW_sspa = norm(y_sspa - gt, 'fro') / norm(gt,'fro') * 100;

% ------------ Console summary -------------------------------------------
clc
disp('---------------------------- Urban (full image) ----------------------------')
fprintf('CSSNMF: Rel. Frob = %6.4e | SSIM = %5.3f | dW%% = %6.3f\n', res_css,  ssim_css,  dW_css);
fprintf('SPA   : Rel. Frob = %6.4e | SSIM = %5.3f | dW%% = %6.3f\n', res_spa,  ssim_spa,  dW_spa);
fprintf('SSPA  : Rel. Frob = %6.4e | SSIM = %5.3f | dW%% = %6.3f | nplp = %d\n', ...
        res_sspa, ssim_sspa, dW_sspa, nplp_best);
disp('---------------------------------------------------------------------------')

% ------------ (Optional) SSIM per map ------------------------------------
term_css  = zeros(1,r);
term_spa  = zeros(1,r);
term_sspa = zeros(1,r);
for t=1:r
    term_css(t)  = ssim(reshape(H_css_re(t,: )/max(H_css_re(t,: )),  [307 307]), ...
                        reshape(A(t,:)/max(A(t,:)),               [307 307]));
    term_spa(t)  = ssim(reshape(H_spa_re(t,: )/max(H_spa_re(t,: )),  [307 307]), ...
                        reshape(A(t,:)/max(A(t,:)),               [307 307]));
    term_sspa(t) = ssim(reshape(H_sspa_re(t,:)/max(H_sspa_re(t,:)),  [307 307]), ...
                        reshape(A(t,:)/max(A(t,:)),               [307 307]));
end
disp('SSIM per material (CSSNMF):'); disp(term_css);  fprintf('Mean: %5.3f\n', mean(term_css));
disp('SSIM per material (SPA)   :'); disp(term_spa);  fprintf('Mean: %5.3f\n', mean(term_spa));
disp('SSIM per material (SSPA)  :'); disp(term_sspa); fprintf('Mean: %5.3f\n', mean(term_sspa));

