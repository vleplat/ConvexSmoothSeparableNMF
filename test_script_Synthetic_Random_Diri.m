addpath(genpath(pwd));
clc
clear

%%-------------------------------------------------------------------------
%% Test 1 (Dirichlet mixtures)
%%-------------------------------------------------------------------------
n0 = 50;
m  = 30;
r  = 5;

%% noise levels and trials
epsilon = logspace(-5,-0.05,7);
iter    = 20;

%% metrics containers
d = length(epsilon);
res_fgnsr    = zeros(iter,d); err_fgnsr    = zeros(iter,d); distW_fgnsr    = zeros(iter,d);
res_spa      = zeros(iter,d); err_spa      = zeros(iter,d); distW_spa      = zeros(iter,d);
res_alg1     = zeros(iter,d); err_alg1     = zeros(iter,d); distW_alg1     = zeros(iter,d);
% SSPA variants
res_sspa_min = zeros(iter,d); err_sspa_min = zeros(iter,d); distW_sspa_min = zeros(iter,d);
res_sspa_mid = zeros(iter,d); err_sspa_mid = zeros(iter,d); distW_sspa_mid = zeros(iter,d);
res_sspa_mean= zeros(iter,d); err_sspa_mean= zeros(iter,d); distW_sspa_mean= zeros(iter,d);

%% settings
flag_noise  = 1;
choice_norm = 2;   % 0: prior H L1 (legacy), 1: prior W,H L1, 2: posterior M L1 (default)

%% trials
for j = 1:iter

    %% Data generation: M = W[H0 H1] with H1 ~ Dirichlet
    H0  = PB(n0, r, 1)';                           % r x n0 one-hot
    n1  = 50; 
    n   = n0 + n1;
    H1  = sample_dirichlet(0.1*ones(r,1), n1)';    % r x n1
    H   = [H0, H1];
    W   = rand(m, r);

    if choice_norm == 1
        H = H ./ sum(H,1);
        W = W ./ sum(W,1);
    end

    M0 = W * H; 
    flag_onmf = 0;
    U  = H ./ sum(H,1);
    M  = M0;

    %% Post-processing options (Algorithm 2)
    options.delta      = 0.95;   % selector threshold for K from X
    options.type       = 1;      % spectral clustering variant
    options.modeltype  = 1 - flag_onmf;  % 1: NNLS (mixed), 0: alternating ONMF
    options.agregation = 0;      % 0: average, 1: median
    options.clustering = 0;      % 0: spectral, 1: kmeans

    %% --- SSPA nplp variants from H0 class sizes ------------------------
    p           = sum(H0 > 0, 2);                  % class sizes p_t
    nplp_min    = max(1, min(p));                  % min_t p_t
    nplp_mean   = max(1, round(mean(p)));          % round(mean_t p_t)
    nplp_mid    = max(1, round(0.5*(nplp_min + nplp_mean)));  % midpoint

    for i = 1:d
        %-------------------------- Noise --------------------------------
        if flag_noise
            Noise = randn(m,n);
            Noise = epsilon(i) * (Noise / norm(Noise,'fro')) * norm(M0,'fro');
            M     = max(M0 + Noise, 0);
        end

        %---------------- Posterior column-wise L1 norm -------------------
        if choice_norm == 2
            M = M ./ sum(M,1);
        end

        %=========================== Alg.1 ================================
        [X_alg1, ~]                 = fgnsr_alg1(M, r, 'maxiter', 1000);
        [W_alg1, H_alg1, ~, ~]      = alg2(M, X_alg1, r, options);
        res_alg1(j,i)               = norm(M - W_alg1*H_alg1, 'fro') / norm(M,'fro');
        H_alg1                      = matchCol(H_alg1', U', W')';
        [err_alg1(j,i), ~]          = Compare_clustering(H_alg1(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);
        W_alg1_av_re                = matchCol(W_alg1, W);
        distW_alg1(j,i)             = norm(W_alg1_av_re./sum(W_alg1_av_re,1) - W./sum(W,1), 'fro') / ...
                                      norm(W./sum(W,1), 'fro');

        %=========================== FGNSR ================================
        [X_fgnsr, K_fgnsr]          = fgnsr(M, r, 'maxiter', 1000);
        W_fgnsr                     = M(:, K_fgnsr);
        H_fgnsr                     = nnlsHALSupdt_new(W_fgnsr'*M, W_fgnsr, [], 1000);
        res_fgnsr(j,i)              = norm(M - W_fgnsr*H_fgnsr, 'fro') / norm(M,'fro');
        H_fgnsr                     = matchCol(H_fgnsr', U', W')';
        [err_fgnsr(j,i), ~]         = Compare_clustering(H_fgnsr(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);
        W_fgnsr_re                  = matchCol(W_fgnsr, W);
        distW_fgnsr(j,i)            = norm(W_fgnsr_re./sum(W_fgnsr_re,1) - W./sum(W,1), 'fro') / ...
                                      norm(W./sum(W,1), 'fro');

        %============================= SPA ===============================
        K_spa                       = FastSepNMF(M, r, 0);
        W_spa                       = M(:, K_spa);
        H_spa                       = nnlsHALSupdt_new(W_spa'*M, W_spa, [], 1000);
        res_spa(j,i)                = norm(M - W_spa*H_spa, 'fro') / norm(M,'fro');
        H_spa                       = matchCol(H_spa', U', W')';
        [err_spa(j,i), ~]           = Compare_clustering(H_spa(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);
        W_spa_re                    = matchCol(W_spa, W);
        distW_spa(j,i)              = norm(W_spa_re./sum(W_spa_re,1) - W./sum(W,1), 'fro') / ...
                                      norm(W./sum(W,1), 'fro');

        %============================= SSPA ==============================
        % --- min variant
        [W_sspa_min, ~]             = SSPA(M, r, nplp_min);
        H_sspa_min                  = nnlsHALSupdt_new(W_sspa_min'*M, W_sspa_min, [], 1000);
        res_sspa_min(j,i)           = norm(M - W_sspa_min*H_sspa_min, 'fro') / norm(M,'fro');
        H_sspa_min                  = matchCol(H_sspa_min', U', W')';
        [err_sspa_min(j,i), ~]      = Compare_clustering(H_sspa_min(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);
        W_sspa_min_re               = matchCol(W_sspa_min, W);
        distW_sspa_min(j,i)         = norm(W_sspa_min_re./sum(W_sspa_min_re,1) - W./sum(W,1), 'fro') / ...
                                      norm(W./sum(W,1), 'fro');

        % --- mid variant
        [W_sspa_mid, ~]             = SSPA(M, r, nplp_mid);
        H_sspa_mid                  = nnlsHALSupdt_new(W_sspa_mid'*M, W_sspa_mid, [], 1000);
        res_sspa_mid(j,i)           = norm(M - W_sspa_mid*H_sspa_mid, 'fro') / norm(M,'fro');
        H_sspa_mid                  = matchCol(H_sspa_mid', U', W')';
        [err_sspa_mid(j,i), ~]      = Compare_clustering(H_sspa_mid(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);
        W_sspa_mid_re               = matchCol(W_sspa_mid, W);
        distW_sspa_mid(j,i)         = norm(W_sspa_mid_re./sum(W_sspa_mid_re,1) - W./sum(W,1), 'fro') / ...
                                      norm(W./sum(W,1), 'fro');

        % --- mean variant
        [W_sspa_mean, ~]            = SSPA(M, r, nplp_mean);
        H_sspa_mean                 = nnlsHALSupdt_new(W_sspa_mean'*M, W_sspa_mean, [], 1000);
        res_sspa_mean(j,i)          = norm(M - W_sspa_mean*H_sspa_mean, 'fro') / norm(M,'fro');
        H_sspa_mean                 = matchCol(H_sspa_mean', U', W')';
        [err_sspa_mean(j,i), ~]     = Compare_clustering(H_sspa_mean(:,1:n0)', U(:,1:n0)', 0, ~flag_onmf);
        W_sspa_mean_re              = matchCol(W_sspa_mean, W);
        distW_sspa_mean(j,i)        = norm(W_sspa_mean_re./sum(W_sspa_mean_re,1) - W./sum(W,1), 'fro') / ...
                                      norm(W./sum(W,1), 'fro');

        %---------------------- quick console log -------------------------
        disp('------------------------------------------------------------------------------------------');
        disp('---- Alg.1 | FGNSR | SPA | SSPA(min) | SSPA(mid) | SSPA(mean) ---------------------------');
        fprintf('Rel. Frob Err : %2.4e | %2.4e | %2.4e | %2.4e | %2.4e | %2.4e\n', ...
            res_alg1(j,i), res_fgnsr(j,i), res_spa(j,i), res_sspa_min(j,i), res_sspa_mid(j,i), res_sspa_mean(j,i));
        fprintf('Clustering Err: %2.4e | %2.4e | %2.4e | %2.4e | %2.4e | %2.4e\n', ...
            err_alg1(j,i), err_fgnsr(j,i), err_spa(j,i), err_sspa_min(j,i), err_sspa_mid(j,i), err_sspa_mean(j,i));
        fprintf('Rel. d(W#,W)  : %2.4e | %2.4e | %2.4e | %2.4e | %2.4e | %2.4e\n', ...
            distW_alg1(j,i), distW_fgnsr(j,i), distW_spa(j,i), distW_sspa_min(j,i), distW_sspa_mid(j,i), distW_sspa_mean(j,i));
        disp('------------------------------------------------------------------------------------------');

    end
end

%%-------------------------------------------------------------------------
%% Post-processing
%%-------------------------------------------------------------------------
yourFolder = 'Outputs_script';
if ~isfolder(yourFolder), mkdir(yourFolder); end
close all

font_size = 20;

labels = {'FGNSR','SPA','Alg.1','SSPA(min)','SSPA(mid)','SSPA(mean)'};

%% Average plots - Relative Frobenius errors
fig(1) = figure;
errorbar(epsilon,mean(res_fgnsr,1),   std(res_fgnsr,1),   '-x','LineWidth',2); hold on
errorbar(epsilon,mean(res_spa,1),     std(res_spa,1),     '-*','LineWidth',2);
errorbar(epsilon,mean(res_alg1,1),    std(res_alg1,1),    '-', 'LineWidth',2);
errorbar(epsilon,mean(res_sspa_min,1),std(res_sspa_min,1),'-.','LineWidth',2);
errorbar(epsilon,mean(res_sspa_mid,1),std(res_sspa_mid,1),':', 'LineWidth',2);
errorbar(epsilon,mean(res_sspa_mean,1),std(res_sspa_mean,1),'--','LineWidth',2);
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| M - WH \|_F}{\| M \|_F}$','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','vertical','Interpreter','latex','FontSize',font_size)
title('Average plots - Relative Frobenius Errors','Interpreter','latex','FontSize',font_size)
grid on; set(gca,'XScale','log');
savefig(fig(1), fullfile(yourFolder,"Aver_RelFro.fig"))

%% Average plots - Accuracy
fig(2) = figure;
errorbar(epsilon,mean(1-err_fgnsr,1),   std(1-err_fgnsr,1),   '-x','LineWidth',2); hold on
errorbar(epsilon,mean(1-err_spa,1),     std(1-err_spa,1),     '-*','LineWidth',2);
errorbar(epsilon,mean(1-err_alg1,1),    std(1-err_alg1,1),    '-', 'LineWidth',2);
errorbar(epsilon,mean(1-err_sspa_min,1),std(1-err_sspa_min,1),'-.','LineWidth',2);
errorbar(epsilon,mean(1-err_sspa_mid,1),std(1-err_sspa_mid,1),':', 'LineWidth',2);
errorbar(epsilon,mean(1-err_sspa_mean,1),std(1-err_sspa_mean,1),'--','LineWidth',2);
ylim([0 1])
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('Accuracy','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','vertical','Interpreter','latex','FontSize',font_size)
title('Average plots - Accuracy','Interpreter','latex','FontSize',font_size)
grid on; set(gca,'XScale','log');
savefig(fig(2), fullfile(yourFolder,"Aver_Acc.fig"))

%% Average plots - Relative distance ||W# - W||_F / ||W#||_F
fig(3) = figure;
semilogx(epsilon,mean(distW_fgnsr,1),    '-x','LineWidth',2); hold on
semilogx(epsilon,mean(distW_spa,1),      '-*','LineWidth',2);
semilogx(epsilon,mean(distW_alg1,1),     '-', 'LineWidth',2);
semilogx(epsilon,mean(distW_sspa_min,1), '-.','LineWidth',2);
semilogx(epsilon,mean(distW_sspa_mid,1), ':', 'LineWidth',2);
semilogx(epsilon,mean(distW_sspa_mean,1),'--','LineWidth',2);
ylim([0 1])
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| W^\# - W \|_F}{\| W^\# \|_F}$','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','vertical','Interpreter','latex','FontSize',font_size)
title('Average plots - Relative distance w.r.t. $W^\#$','Interpreter','latex','FontSize',font_size)
grid on;
savefig(fig(3), fullfile(yourFolder,"Aver_distW.fig"))

%% Best among trials - Relative Frobenius errors
fig(4) = figure;
semilogx(epsilon,min(res_fgnsr),    '-x','LineWidth',2); hold on
semilogx(epsilon,min(res_spa),      '-*','LineWidth',2);
semilogx(epsilon,min(res_alg1),     '-', 'LineWidth',2);
semilogx(epsilon,min(res_sspa_min), '-.','LineWidth',2);
semilogx(epsilon,min(res_sspa_mid), ':', 'LineWidth',2);
semilogx(epsilon,min(res_sspa_mean),'--','LineWidth',2);
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| M - WH \|_F}{\| M \|_F}$','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','vertical','Interpreter','latex','FontSize',font_size)
title('Best among trials - Relative Frobenius Errors','Interpreter','latex','FontSize',font_size)
grid on; set(gca,'XScale','log');
savefig(fig(4), fullfile(yourFolder,"Best_RelFro.fig"))

%% Best among trials - Accuracy
fig(5) = figure;
semilogx(epsilon,max(1-err_fgnsr),    '-x','LineWidth',2); hold on
semilogx(epsilon,max(1-err_spa),      '-*','LineWidth',2);
semilogx(epsilon,max(1-err_alg1),     '-', 'LineWidth',2);
semilogx(epsilon,max(1-err_sspa_min), '-.','LineWidth',2);
semilogx(epsilon,max(1-err_sspa_mid), ':', 'LineWidth',2);
semilogx(epsilon,max(1-err_sspa_mean),'--','LineWidth',2);
ylim([0 1])
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('Accuracy','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','vertical','Interpreter','latex','FontSize',font_size)
title('Best among trials - Accuracy','Interpreter','latex','FontSize',font_size)
grid on; set(gca,'XScale','log');
savefig(fig(5), fullfile(yourFolder,"Best_Acc.fig"))

%% Best among trials - Relative distance ||W# - W||_F / ||W#||_F
fig(6) = figure;
semilogx(epsilon,min(distW_fgnsr),    '-x','LineWidth',2); hold on
semilogx(epsilon,min(distW_spa),      '-*','LineWidth',2);
semilogx(epsilon,min(distW_alg1),     '-', 'LineWidth',2);
semilogx(epsilon,min(distW_sspa_min), '-.','LineWidth',2);
semilogx(epsilon,min(distW_sspa_mid), ':', 'LineWidth',2);
semilogx(epsilon,min(distW_sspa_mean),'--','LineWidth',2);
ylim([0 1])
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| W^\# - W \|_F}{\| W^\# \|_F}$','Interpreter','latex','FontSize',font_size);
legend(labels,'Location','northwest','Orientation','vertical','Interpreter','latex','FontSize',font_size)
title('Best among trials - Relative distance w.r.t. $W^\#$','Interpreter','latex','FontSize',font_size)
grid on;
savefig(fig(6), fullfile(yourFolder,"Best_distW.fig"))
