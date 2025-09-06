addpath(genpath(pwd));
clc
clear
disp('----------------------------------------------------------------------------------------------------------')
disp("HSI test - Samson data set - Paper Section 6 ")
disp("The computation of X can take approximatively 30 minutes on a recent laptop...")
disp("You can directly load the .mat ./Results/HSI/Samson and start playing with the code starting from line 56")
disp('----------------------------------------------------------------------------------------------------------')

%%-------------------------------------------------------------------------
%% Load the data sets
%%-------------------------------------------------------------------------
% From https://lesun.weebly.com/hyperspectral-data-set.html
% Load the Ground-truth (W,H)
load('end3.mat') ;
W_true = M;
H_true = A;
clear M A B

% Load the HSI
load('samson_1.mat') ;
M = V;
clear V;

% Normalisation of M 
M=M./sum(M);

% Number of cluster/the factorization rank r
r = 3;

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
[X_fgnsr, K_fgnsr_1] = fgnsr_alg1(M, r, 'maxiter', 200, 'mu',0.5);

%% ------------------------------------------------------------------------
%% Samson — CSSNMF vs SPA vs SSPA (same flow as Jasper)
%% ------------------------------------------------------------------------

% ---- Choose delta via quick histogram (same as Jasper) -------------------
close all;
histogram(sum(X_fgnsr,2));

% ---- Postprocessing options (same style as Jasper) -----------------------
options.delta      = 1.08;   % dataset-specific; keep your good value
options.agregation = 1;      % 1: median (as in Jasper); 0: average
options.clustering = 1;      % 1: kmeans (as in Jasper); 0: spectral
options.type       = 1;      % spectral clustering variant id
options.modeltype  = 1;      % NNLS

% ---- CSSNMF (Alg. 1 + 2) -------------------------------------------------
[W_fgnsr, H_fgnsr, K_fgnsr, ~] = alg2(M, X_fgnsr, r, options);
res_fgnsr = norm(M - W_fgnsr*H_fgnsr, 'fro') / norm(M, 'fro');
%%
% ---- SPA baseline (pure SPA) ---------------------------------------------
K_spa_pure  = FastSepNMF(M, r, 0);
W_spa_pure  = M(:, K_spa_pure);
H_spa_pure  = nnlsHALSupdt_new(W_spa_pure' * M, W_spa_pure, [], 1000);
res_spa_pure = norm(M - W_spa_pure*H_spa_pure, 'fro') / norm(M, 'fro');

% ---- SSPA: pick nplp by quick sweep (same idea as Jasper) ----------------
res_sspa_sweep = [];
for nplp_test = 10:50:2000
    [W_tmp, ~]  = SSPA(M, r, nplp_test);
    H_tmp       = nnlsHALSupdt_new(W_tmp' * M, W_tmp, [], 1000);
    res_sspa_sweep(end+1) = norm(M - W_tmp*H_tmp, 'fro') / norm(M, 'fro'); 
end
figure; plot(10:50:2000, res_sspa_sweep, '-o'); grid on;
xlabel('nplp'); ylabel('Rel. Frob. Error');
title('SSPA nplp sweep (Samson)');

% ----- Choose your best nplp from the sweep (example below) ---------------
nplp = 1200;             % <- keep your tuned value
Options.average = 0;     % 0: median (default), 1: mean
[W_sspa, ~]  = SSPA(M, r, nplp, Options);
H_sspa       = nnlsHALSupdt_new(W_sspa' * M, W_sspa, [], 1000);
res_sspa     = norm(M - W_sspa*H_sspa, 'fro') / norm(M, 'fro');

%% ------------------------------------------------------------------------
%% Align to ground truth (W_true, H_true) and compute metrics
%% ------------------------------------------------------------------------
A = H_true;                   % GT abundances (r x N)
B = W_true;                   % GT endmembers (m x r)

% H alignment (columns are pixels)
H_cssnmf_re   = matchCol(H_fgnsr', A')';
H_spa_pure_re = matchCol(H_spa_pure', A')';
H_sspa_re     = matchCol(H_sspa', A')';

% W alignment
W_cssnmf_re   = matchCol(W_fgnsr, B);
W_spa_pure_re = matchCol(W_spa_pure, B);
W_sspa_re     = matchCol(W_sspa, B);

% Relative W-errors (scale-invariant via column normalization)
normc = @(X) X ./ sum(X,1);
dW_cssnmf   = norm( normc(W_cssnmf_re) - normc(B), 'fro') / norm(normc(B),'fro');
dW_spa_pure = norm( normc(W_spa_pure_re) - normc(B), 'fro') / norm(normc(B),'fro');
dW_sspa     = norm( normc(W_sspa_re)     - normc(B), 'fro') / norm(normc(B),'fro');

% Global SSIM (same choice as Jasper: no extra row-wise scaling)
ssim_CSSNMF   = ssim(H_cssnmf_re,   A);
ssim_SPA_pure = ssim(H_spa_pure_re, A);
ssim_SSPA     = ssim(H_sspa_re,     A);

% Per-map SSIM (each row reshaped to 95×95)
term_cssnmf = zeros(1,r);
term_spa    = zeros(1,r);
term_sspa   = zeros(1,r);
for t = 1:r
    gt_map   = reshape(A(t,:)/max(A(t,:)),                 [95 95]);
    f_map    = reshape(H_cssnmf_re(t,:)/max(H_cssnmf_re(t,:)), [95 95]);
    s_map    = reshape(H_spa_pure_re(t,:)/max(H_spa_pure_re(t,:)), [95 95]);
    p_map    = reshape(H_sspa_re(t,:)/max(H_sspa_re(t,:)),  [95 95]);
    term_cssnmf(t) = ssim(f_map, gt_map);
    term_spa(t)    = ssim(s_map, gt_map);
    term_sspa(t)   = ssim(p_map, gt_map);
end

%% ------------------------------------------------------------------------
%% Abundance maps — save EPS (GT, CSSNMF, SPA, SSPA)
%% ------------------------------------------------------------------------
outDir = 'figs_numTests_HSI_Samson'; if ~isfolder(outDir), mkdir(outDir); end
close all;

% Display via your helper and save each as EPS
affichage(A',            r, 95, 95);      title('GT (Samson)');
print('-depsc2', fullfile(outDir, 'Abundances_GT.eps'));      close;

affichage(H_cssnmf_re',  r, 95, 95);      title('CSSNMF');
print('-depsc2', fullfile(outDir, 'Abundances_cssnmf.eps'));  close;

affichage(H_spa_pure_re',r, 95, 95);      title('SPA');
print('-depsc2', fullfile(outDir, 'Abundances_SPA.eps'));     close;

affichage(H_sspa_re',    r, 95, 95);      title('SSPA');
print('-depsc2', fullfile(outDir, 'Abundances_SSPA.eps'));    close;

%% ------------------------------------------------------------------------
%% Signatures panel (2×2): GT | CSSNMF ; SPA | SSPA  -> EPS
%% ------------------------------------------------------------------------
x = 1:size(M,1);
colors    = {'b','k','r','g','c','m'};
linestyle = {'-','--','-.',':'};
getFirst  = @(v)v{1};
getprop   = @(opts, idx)getFirst(circshift(opts, -idx+1));
linew     = 1.5; fontSize = 13;

figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);

% GT
subplot(2,2,1);
hold on;
for t=1:r
    gt(:,t) = B(:,t)/sum(B(:,t));
    plot(x, gt(:,t), 'Color', getprop(colors,t), 'LineStyle', getprop(linestyle,t), 'LineWidth', linew);
end
xlim([1, numel(x)]); grid on; title('Ground truth','Interpreter','latex','FontSize',fontSize);

% CSSNMF
subplot(2,2,2);
hold on;
for t=1:r
    y_css(:,t) = W_cssnmf_re(:,t)/sum(W_cssnmf_re(:,t));
    plot(x, y_css(:,t), 'Color', getprop(colors,t), 'LineStyle', getprop(linestyle,t), 'LineWidth', linew);
end
xlim([1, numel(x)]); grid on; title('CSSNMF','Interpreter','latex','FontSize',fontSize);

% SPA
subplot(2,2,3);
hold on;
for t=1:r
    y_spa(:,t) = W_spa_pure_re(:,t)/sum(W_spa_pure_re(:,t));
    plot(x, y_spa(:,t), 'Color', getprop(colors,t), 'LineStyle', getprop(linestyle,t), 'LineWidth', linew);
end
xlim([1, numel(x)]); grid on; title('SPA','Interpreter','latex','FontSize',fontSize);

% SSPA
subplot(2,2,4);
hold on;
for t=1:r
    y_sspa(:,t) = W_sspa_re(:,t)/sum(W_sspa_re(:,t));
    plot(x, y_sspa(:,t), 'Color', getprop(colors,t), 'LineStyle', getprop(linestyle,t), 'LineWidth', linew);
end
xlim([1, numel(x)]); grid on; title(sprintf('SSPA (nplp=%d)', nplp),'Interpreter','latex','FontSize',fontSize);

sgtitle('Endmember spectral signatures — Samson','Interpreter','latex');
print('-depsc2', fullfile(outDir, 'signatures_grid.eps'));
close;

%% ------------------------------------------------------------------------
%% Console summary (same style as Jasper)
%% ------------------------------------------------------------------------
clc;
disp('---------------------  CSSNMF  |    SPA     |    SSPA   ---------------------');
fprintf('Rel. Frob Error (lower is better):  %8.4e | %8.4e | %8.4e\n', res_fgnsr, res_spa_pure, res_sspa);
fprintf('SSIM (1 best):                      %8.4f | %8.4f | %8.4f\n', ssim_CSSNMF, ssim_SPA_pure, ssim_SSPA);
fprintf('Rel. W-error (||W#-W||/||W#||):     %8.4e | %8.4e | %8.4e\n', dW_cssnmf, dW_spa_pure, dW_sspa);
disp('Per-map SSIM (mean over materials):');
fprintf('  CSSNMF: %.4f   SPA: %.4f   SSPA: %.4f\n', mean(term_cssnmf), mean(term_spa), mean(term_sspa));
disp('-----------------------------------------------------------------------------');
