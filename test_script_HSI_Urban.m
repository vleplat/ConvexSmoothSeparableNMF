addpath(genpath(pwd));
clc
clear
disp('----------------------------------------------------------------------------------------------------------')
disp("HSI test - Urban data set - Paper Section 6 ")
disp("The computation of X can take approximatively 45 minutes on a recent laptop...")
disp("You can directly load the .mat ./Results/HSI/Urban and start playing with the code starting from line 55")
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

%% 
% Help chosing a value for delta
close all
histogram(summ(X_fgnsr,2));
%% Selection of best delta
res_fgnsr_s_list = [];
res_SSIM_s_list = [];
for delta=0.2:0.1:1.8
    options.delta = delta;
    [W_fgnsr_s,H_fgnsr_s,K_fgnsr_s,Wfgnsr_s] = alg2(Ms,X_fgnsr,r,options);
    res_fgnsr_s_list = [res_fgnsr_s_list norm(Ms-W_fgnsr_s*H_fgnsr_s,'fro')./norm(Ms,'fro')];
    H_fgnsr=nnlsHALSupdt_new(W_fgnsr_s'*M,W_fgnsr_s,[],1000);
    H_fgnsr_re= matchCol(H_fgnsr',A')';
    res_SSIM_s_list = [res_SSIM_s_list ssim(H_fgnsr_re,A);]
end
close all;
figure;
plot(0.2:0.1:1.8,res_fgnsr_s_list);
grid on
xlabel('delta')
ylabel('Rel. Frob. Error.')

close all;
figure;
plot(0.2:0.1:1.8,res_SSIM_s_list);
grid on
xlabel('delta')
ylabel('SSIM')

%%
options.delta=1.04;  %SSIM 7.94 for 1.4 for 200 iterations | 1.1 for 400 iterations
options.agregation = 0;
options.clustering = 0;
[W_fgnsr_s,H_fgnsr_s,K_fgnsr_s,Wfgnsr_s] = alg2(Ms,X_fgnsr,r,options);

% compute the Relative Frobenius Error - Sampled data set
res_fgnsr_s = norm(Ms-W_fgnsr_s*H_fgnsr_s,'fro')./norm(Ms,'fro')

% compute the Relative Frobenius Error - Full data set
W_fgnsr = W_fgnsr_s;
H_fgnsr=nnlsHALSupdt_new(W_fgnsr'*M,W_fgnsr,[],1000);
res_fgnsr = norm(M-W_fgnsr_s*H_fgnsr,'fro')./norm(M,'fro')

%% SPA/SSPA
clc
%% Selection of best nplp
res_spa_s_list = [];
for nplp=10:50:1000
    [W_spa,K_spa] = SSPA(Ms, r, nplp);
    H_spas=nnlsHALSupdt_new(W_spa'*Ms,W_spa,[],1000);
    res_spa_s_list = [res_spa_s_list norm(Ms-W_spa*H_spas,'fro')./norm(Ms,'fro')];
end
% Display error w.r.t. nplp values
figure;
plot(10:50:1000,res_spa_s_list);
grid on
xlabel('nplp')
ylabel('Rel. Frob. Error.')
% Conclusion: lowest value reached for nplp = 110; -> based on abundance
% maps, nplp=100 seems to be the best
%% Final run for SSPA
% Sampled data 
nplp = 100;  % 
Options.average = 0; % 1 mean , 0 median (default)
[W_spa,K_spa] = SSPA(Ms, r, nplp);
H_spas=nnlsHALSupdt_new(W_spa'*Ms,W_spa,[],1000);

% compute the Relative Frobenius Error - Sampled data set
res_spa_s = norm(Ms-W_spa*H_spas,'fro')./norm(Ms,'fro')

% % compute the Relative Frobenius Error - Full data set
H_spa=nnlsHALSupdt_new(W_spa'*M,W_spa,[],1000);
res_spa=norm(M-W_spa*H_spa,'fro')./norm(M,'fro')

%% -------------------------------------------------------------------------
%%Display some results
%%-------------------------------------------------------------------------
%% Abundance maps - full data set
close all
A = H_true;
H_fgnsr_re= matchCol(H_fgnsr',A')';
H_spa_re= matchCol(H_spa',A')';

affichage((diag(1./max(H_true,[],2))*H_true)',6,307,307);              %groudtruth 
affichage((diag(1./max(H_fgnsr_re,[],2))*H_fgnsr_re)',6,307,307);      %estimated CSSNMF
affichage((diag(1./max(H_spa_re,[],2))*H_spa_re)',6,307,307);          %estimated SSPA

%% Spectral signatures
close all
B = W_true;
W_fgnsr_re= matchCol(W_fgnsr,W_true);
W_spa_re= matchCol(W_spa,W_true);
x=1:162; 

% https://ch.mathworks.com/matlabcentral/answers/697655-how-can-i-plot-with-different-markers-linestyles-and-colors-in-a-loop
markers = {'.','+','*','.','x','_','.','.'};
colors = {'b','k','r','g','c','m'};
linestyle = {'-','--','-.',':','-'};

getFirst = @(v)v{1}; 
getprop = @(options, idx)getFirst(circshift(options,-idx+1));
linew = 1.5;
fontSize = 14;
figure
subplot(1,2,1) %groudtruth 
for t=1:r
    gt(:,t)=B(:,t)/sum(B(:,t)); hold on 
    plot(x,gt(:,t),'Marker',getprop(markers,t),'color',getprop(colors,t),'linestyle',getprop(linestyle,t),'LineWidth',linew)
    % axis([0 162 0 1])
    set(gca,'xlim',[1 162]);
end
title('Ground-truth','Interpreter','latex','FontSize',fontSize); 
legend('Asphalt Road','Grass', 'Tree','Roof','Metal','Dirt','Interpreter','latex','FontSize',fontSize);  
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize); 
grid on
 
subplot(1,2,2)
for t=1:r
    y(:,t)=W_fgnsr_re(:,t)/sum(W_fgnsr_re(:,t));
    y_spa(:,t)=W_spa_re(:,t)/sum(W_spa_re(:,t));
    hold on
    plot(x,y(:,t),'Marker',getprop(markers,t),'color',getprop(colors,t),'linestyle',getprop(linestyle,t),'LineWidth',linew)
    % axis([0 162 0 1])
    set(gca,'xlim',[1 162]);
end
title('CSSNMF','Interpreter','latex','FontSize',fontSize); 
legend('Asphalt Road','Grass', 'Tree','Roof','Metal','Dirt','Interpreter','latex','FontSize',fontSize); 
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize); 
grid on

%% Compute SSIM - globally
ssim_CSSNMF=ssim(H_fgnsr_re,A);
ssim_SSPA=ssim(H_spa_re,A);

% ssim_CSSNMF=ssim(diag(1./max(H_fgnsr_re,[],2))*H_fgnsr_re,diag(1./max(A,[],2))*A);
% ssim_SSPA=ssim(diag(1./max(H_spa_re,[],2))*H_spa_re,diag(1./max(A,[],2))*A);


%% Display Rel. Frob. Errors and SSIM
clc
disp('------------------------------------      CSSNMF  |    SSPA    -----------------')
fprintf('Rel. Frob Error (lower the better):     %2.4e|  %2.4e  \n', res_fgnsr,res_spa);
fprintf('SSIM (1 best):                           %2.2e |  %2.2e  \n', ssim_CSSNMF,ssim_SSPA);
fprintf('||W-W#||_F/||W#||_F (lower the better):  %2.2e |  %2.2e  \n', norm(y-gt,'fro')/norm(gt,'fro')*100,norm(y_spa-gt,'fro')/norm(gt,'fro')*100);
disp('----------------------------------------------------------------------------------------------------------')
  
%% Compute SSIM - per map
term_cssnmf = [];
for t=1:r
    term_cssnmf = [term_cssnmf ssim(reshape(H_fgnsr_re(t,:)/max(H_fgnsr_re(t,:)),[307 307]),reshape(A(t,:)/max(A(t,:)),[307 307]))];
end
disp('----------------- CSSNMF --------------------')
term_cssnmf
mean(term_cssnmf)
disp('---------------------------------------------')
term_sspa = [];
for t=1:r
    term_sspa = [term_sspa ssim(reshape(H_spa_re(t,:)/max(H_spa_re(t,:)),[307 307]),reshape(A(t,:)/max(A(t,:)),[307 307]))];
end
disp('----------------- SSPA --------------------')
term_sspa
mean(term_sspa)
disp('---------------------------------------------')
