addpath(genpath(pwd));
clc; clear;
rng(2025);

%% ------------------------------------------------------------------------
%% Middle points + adversarial noise (modular in r and n1)
%% ------------------------------------------------------------------------

% Core dimensions
m  = 30;        % ambient dimension
r  = 10;         % number of endmembers / materials
n0 = 50;        % number of pure (one-hot) columns

% Choose how to build middle points (H1)
midpoint_mode = 'all';  % 'all' (use all nchoosek(r,2) pairs) or 'sampled'
n1_requested  = 100;     % if 'sampled', set e.g. n1_requested = 15; otherwise [] is ignored

% Noise sweep and trials
epsilon = logspace(-3, -0.05, 4);   % noise levels
iter    = 10;                       % number of trials
d       = numel(epsilon);

% Normalization choice
% 0: prior L1 normalization of H only (legacy ONMF assumption)  [rarely used]
% 1: prior L1 normalization of both W and H (noise added after)  [use with care]
% 2: posterior column-wise L1 normalization of M                 [default]
choice_norm = 2;

% Metrics containers
res_fgnsr   = zeros(iter,d); err_fgnsr   = zeros(iter,d); distW_fgnsr   = zeros(iter,d);
res_spa     = zeros(iter,d); err_spa     = zeros(iter,d); distW_spa     = zeros(iter,d);
res_alg1    = zeros(iter,d); err_alg1    = zeros(iter,d); distW_alg1    = zeros(iter,d);
res_sspa    = zeros(iter,d); err_sspa    = zeros(iter,d); distW_sspa    = zeros(iter,d);

flag_noise  = 1;           % add adversarial noise on H1 block
flag_onmf   = 0;           % 0: mixed model in postprocessing

% Post-processing options (Algorithm 2)
options.delta      = 0.95;   % threshold for selecting K from X
options.type       = 1;      % spectral clustering variant id
options.modeltype  = 1 - flag_onmf;  % 1: NNLS (mixed), 0: alternating ONMF
options.agregation = 1;      % 0: average, 1: median   (we use median here)
options.clustering = 0;      % 0: spectral, 1: kmeans

for j = 1:iter
    % --- Pure block (H0) and middle-point block (H1) ---------------------
    H0 = PB(n0, r, 1)';   % one-hot columns (size r x n0)
    [H1, n1_used] = build_midpoints_block(r, midpoint_mode, n1_requested);  % r x n1
    H  = [H0, H1];
    n  = n0 + n1_used;

    % Draw W and (optionally) apply prior normalization
    W = rand(m, r);
    if choice_norm == 1
        H = H ./ sum(H,1); 
        W = W ./ sum(W,1);
    elseif choice_norm == 0
        H = H ./ sum(H,1);  % legacy ONMF assumption on H
    end

    M0 = W * H; 
    U  = H ./ sum(H,1);     % column-stochastic version of H, for alignment

    % SSPA: number of proximal latent points ~ average class size
    p = sum(H0 > 0, 2);          % p(t) = # pure columns in class t
    nplp_mean = round(mean(p));  % average class size  (default we used)
    nplp_min  = max(1, min(p));  % conservative choice (never exceeds any class)

    nplp = nplp_mean;            % or set nplp = nplp_min for a stricter setting

    for i = 1:d
        M = M0;  % fresh copy

        % --- Adversarial noise: none on H0, outward shift on H1 ----------
        if flag_noise
            Noise          = zeros(m, n);
            wbar           = mean(W * H0, 2);                           % centroid of pure block
            Noise(:,1:n0)  = 0;                                         % no noise on pure
            Noise(:,n0+1:end) = M0(:,n0+1:end) - wbar * ones(1, n1_used); % outward shift
            % scale so that ||N||_F = eps * ||M0||_F
            Noise = epsilon(i) * (Noise / norm(Noise, 'fro')) * norm(M0, 'fro');
            M     = max(M0 + Noise, 0);
        end

        % --- Posterior normalization of M (default) ----------------------
        if choice_norm == 2
            M = M ./ sum(M,1);
        end

        % =====================  Alg. 1  ==================================
        [X_alg1, ~]             = fgnsr_alg1(M, r, 'maxiter', 1000);
        [W_alg1, H_alg1, ~, ~]  = alg2(M, X_alg1, r, options);

        res_alg1(j,i) = norm(M - W_alg1 * H_alg1, 'fro') / norm(M, 'fro');

        H_alg1        = matchCol(H_alg1', U', W')';
        [err_alg1(j,i), ~] = Compare_clustering(H_alg1(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);

        W_alg1_re     = matchCol(W_alg1, W);
        distW_alg1(j,i)= norm(W_alg1_re./sum(W_alg1_re,1) - W./sum(W,1), 'fro') / ...
                         norm(W./sum(W,1), 'fro');

        % =====================  FGNSR  ===================================
        [X_fgnsr, K_fgnsr] = fgnsr(M, r, 'maxiter', 1000);
        W_fgnsr = M(:, K_fgnsr);
        H_fgnsr = nnlsHALSupdt_new(W_fgnsr' * M, W_fgnsr, [], 1000);

        res_fgnsr(j,i) = norm(M - W_fgnsr * H_fgnsr, 'fro') / norm(M, 'fro');

        H_fgnsr        = matchCol(H_fgnsr', U', W')';
        [err_fgnsr(j,i), ~] = Compare_clustering(H_fgnsr(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);

        W_fgnsr_re     = matchCol(W_fgnsr, W);
        distW_fgnsr(j,i)= norm(W_fgnsr_re./sum(W_fgnsr_re,1) - W./sum(W,1), 'fro') / ...
                          norm(W./sum(W,1), 'fro');

        % =====================  SPA  =====================================
        K_spa  = FastSepNMF(M, r, 0);
        W_spa  = M(:, K_spa);
        H_spa  = nnlsHALSupdt_new(W_spa' * M, W_spa, [], 1000);

        res_spa(j,i)   = norm(M - W_spa * H_spa, 'fro') / norm(M, 'fro');

        H_spa          = matchCol(H_spa', U', W')';
        [err_spa(j,i), ~] = Compare_clustering(H_spa(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);

        W_spa_re       = matchCol(W_spa, W);
        distW_spa(j,i) = norm(W_spa_re./sum(W_spa_re,1) - W./sum(W,1), 'fro') / ...
                         norm(W./sum(W,1), 'fro');

        % =====================  SSPA  ====================================
        [W_sspa, ~] = SSPA(M, r, nplp);
        H_sspa      = nnlsHALSupdt_new(W_sspa' * M, W_sspa, [], 1000);

        res_sspa(j,i) = norm(M - W_sspa * H_sspa, 'fro') / norm(M, 'fro');

        H_sspa         = matchCol(H_sspa', U', W')';
        [err_sspa(j,i), ~] = Compare_clustering(H_sspa(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);

        W_sspa_re      = matchCol(W_sspa, W);
        distW_sspa(j,i)= norm(W_sspa_re./sum(W_sspa_re,1) - W./sum(W,1), 'fro') / ...
                         norm(W./sum(W,1), 'fro');

        % ------------------- quick console log ---------------------------
        disp('--------------------------------------------------------------------');
        disp('-------------------   Alg.1 |   FGNSR   |   SSPA  ------------------');
        fprintf('Rel. Frob Error : %2.4e | %2.4e | %2.4e\n', res_alg1(j,i), res_fgnsr(j,i), res_sspa(j,i));
        fprintf('Clustering Err. : %2.4e | %2.4e | %2.4e\n', err_alg1(j,i), err_fgnsr(j,i), err_sspa(j,i));
        fprintf('Rel. d(W#,W)    : %2.4e | %2.4e | %2.4e\n', distW_alg1(j,i), distW_fgnsr(j,i), distW_sspa(j,i));
        disp('--------------------------------------------------------------------');
    end
end

%% ------------------------------------------------------------------------
%% Post-processing (same visuals as before; only legend text fixed)
%% ------------------------------------------------------------------------
outDir = 'Outputs_script'; if ~isfolder(outDir), mkdir(outDir); end
close all;

font_size = 14; 
labels = {'FGNSR','SPA','Alg.1','SSPA'};

% Average: Relative Frobenius errors
fig(1) = figure;
errorbar(epsilon, mean(res_fgnsr,1), std(res_fgnsr,1), '-x', 'LineWidth',2); hold on;
errorbar(epsilon, mean(res_spa,1),   std(res_spa,1),   '-*', 'LineWidth',2);
errorbar(epsilon, mean(res_alg1,1),  std(res_alg1,1),  '-',  'LineWidth',2);
errorbar(epsilon, mean(res_sspa,1),  std(res_sspa,1),  '-.', 'LineWidth',2);
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\| M - WH \|_F / \| M \|_F$','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','horizontal','Interpreter','latex','FontSize',font_size);
title('Average plots - Relative Frobenius Errors','Interpreter','latex','FontSize',font_size); grid on; set(gca,'XScale','log');
savefig(fig(1), fullfile(outDir,'Aver_RelFro.fig'));

% Average: Accuracy (1 - clustering error)
fig(2) = figure;
errorbar(epsilon, mean(1-err_fgnsr,1), std(1-err_fgnsr,1), '-x','LineWidth',2); hold on;
errorbar(epsilon, mean(1-err_spa,1),   std(1-err_spa,1),   '-*','LineWidth',2);
errorbar(epsilon, mean(1-err_alg1,1),  std(1-err_alg1,1),  '-', 'LineWidth',2);
errorbar(epsilon, mean(1-err_sspa,1),  std(1-err_sspa,1),  '-.','LineWidth',2);
ylim([0 1]);
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('Accuracy','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','horizontal','Interpreter','latex','FontSize',font_size);
title('Average plots - Accuracy','Interpreter','latex','FontSize',font_size); grid on; set(gca,'XScale','log');
savefig(fig(2), fullfile(outDir,'Aver_Acc.fig'));

% Average: Relative distance d(W#, W)
fig(3) = figure;
semilogx(epsilon, mean(distW_fgnsr,1), '-x','LineWidth',2); hold on;
semilogx(epsilon, mean(distW_spa,1),   '-*','LineWidth',2);
semilogx(epsilon, mean(distW_alg1,1),  '-', 'LineWidth',2);
semilogx(epsilon, mean(distW_sspa,1),  '-.','LineWidth',2);
ylim([0 1]);
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\| W^\# - W \|_F / \| W^\# \|_F$','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','horizontal','Interpreter','latex','FontSize',font_size);
title('Average plots - Relative distance w.r.t. $W^\#$','Interpreter','latex','FontSize',font_size); grid on;
savefig(fig(3), fullfile(outDir,'Aver_distW.fig'));

% Best (min or max across trials)
fig(4) = figure;
semilogx(epsilon, min(res_fgnsr), '-x','LineWidth',2); hold on;
semilogx(epsilon, min(res_spa),   '-*','LineWidth',2);
semilogx(epsilon, min(res_alg1),  '-', 'LineWidth',2);
semilogx(epsilon, min(res_sspa),  '-.','LineWidth',2);
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\| M - WH \|_F / \| M \|_F$','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','horizontal','Interpreter','latex','FontSize',font_size);
title('Best among trials - Relative Frobenius Errors','Interpreter','latex','FontSize',font_size); grid on;
savefig(fig(4), fullfile(outDir,'Best_RelFro.fig'));

fig(5) = figure;
semilogx(epsilon, max(1-err_fgnsr), '-x','LineWidth',2); hold on;
semilogx(epsilon, max(1-err_spa),   '-*','LineWidth',2);
semilogx(epsilon, max(1-err_alg1),  '-', 'LineWidth',2);
semilogx(epsilon, max(1-err_sspa),  '-.','LineWidth',2);
ylim([0 1]);
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('Accuracy','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','horizontal','Interpreter','latex','FontSize',font_size);
title('Best among trials - Accuracy','Interpreter','latex','FontSize',font_size); grid on;
savefig(fig(5), fullfile(outDir,'Best_Acc.fig'));

fig(6) = figure;
semilogx(epsilon, min(distW_fgnsr), '-x','LineWidth',2); hold on;
semilogx(epsilon, min(distW_spa),   '-*','LineWidth',2);
semilogx(epsilon, min(distW_alg1),  '-', 'LineWidth',2);
semilogx(epsilon, min(distW_sspa),  '-.','LineWidth',2);
ylim([0 1]);
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\| W^\# - W \|_F / \| W^\# \|_F$','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','horizontal','Interpreter','latex','FontSize',font_size);
title('Best among trials - Relative distance w.r.t. $W^\#$','Interpreter','latex','FontSize',font_size); grid on;
savefig(fig(6), fullfile(outDir,'Best_distW.fig'));

%% ------------------------------------------------------------------------
%% Local helper: build H1 as pairwise midpoints (modular in r and n1)
%% ------------------------------------------------------------------------
function [H1, n1_used] = build_midpoints_block(r, mode, n1_req)
    pairs_all = nchoosek(1:r, 2);     % all unordered pairs
    P = size(pairs_all,1);
    switch lower(mode)
        case 'all'
            pairs_sel = pairs_all;
        case 'sampled'
            if isempty(n1_req), n1_req = min(P, r*(r-1)/2); end
            if n1_req <= P
                idx = randperm(P, n1_req);
                pairs_sel = pairs_all(idx,:);
            else
                % allow repeats if more than all pairs are requested
                idx = randi(P, n1_req, 1);
                pairs_sel = pairs_all(idx,:);
            end
        otherwise
            error('Unknown midpoint_mode: use ''all'' or ''sampled''.');
    end
    n1_used = size(pairs_sel,1);
    H1 = zeros(r, n1_used);
    for k = 1:n1_used
        H1(pairs_sel(k,1), k) = 0.5;
        H1(pairs_sel(k,2), k) = 0.5;
    end
end

