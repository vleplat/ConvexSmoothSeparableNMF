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

%% 
% Help chosing a value for delta
close all
histogram(sum(X_fgnsr,2));
%% 
options.delta=1.15; %1.11 or 1.3 ...
options.agregation = 1;
options.clustering = 1;
[W_fgnsr,H_fgnsr,K_fgnsr,Wfgnsr] = alg2(M,X_fgnsr,r,options);
% compute the Relative Frobenius Error 
res_fgnsr = norm(M-W_fgnsr*H_fgnsr,'fro')./norm(M,'fro')

%% SPA/SSPA

%% Selection of best nplp
res_spa_list = [];
for nplp=10:50:1200
    [W_spa,K_spa] = SSPA(M, r, nplp);
    H_spa=nnlsHALSupdt_new(W_spa'*M,W_spa,[],1000);
    % compute the Relative Frobenius Error 
    res_spa_list = [res_spa_list norm(M-W_spa*H_spa,'fro')./norm(M,'fro')];
end
% Display error w.r.t. nplp values
figure;
plot(10:50:1200,res_spa_list);
grid on
xlabel('nplp')
ylabel('Rel. Frob. Error.')
% Conclusion: lowest value reached for nplp = 910;
%% Run SSPA for nplp computed above
nplp = 910; %910 the best 
Options.average = 0; % 1 mean , 0 median (default)
[W_spa,K_spa] = SSPA(M, r, nplp, Options);
H_spa=nnlsHALSupdt_new(W_spa'*M,W_spa,[],1000);
% compute the Relative Frobenius Error 
res_spa=norm(M-W_spa*H_spa,'fro')./norm(M,'fro')

%%-------------------------------------------------------------------------
%% Display some results
%%-------------------------------------------------------------------------
%% Abundance maps
close all
A = H_true;
H_fgnsr_re= matchCol(H_fgnsr',A')';
H_spa_re= matchCol(H_spa',A')';

affichage(H_true',4,100,100);          %groudtruth 
affichage(H_fgnsr_re',4,100,100);      %estimated CSSNMF
affichage(H_spa_re',4,100,100);        %estimated SSPA


%% Spectral signatures
close all
B = W_true;
W_fgnsr_re= matchCol(W_fgnsr,W_true);
W_spa_re= matchCol(W_spa,W_true);
x=1:198; 

% markers = {'o','+','*','s','d','v','>','h'};
colors = {'b','k','r','g','c','m'};
linestyle = {'-','--','-.',':'};

getFirst = @(v)v{1}; 
getprop = @(options, idx)getFirst(circshift(options,-idx+1));
linew = 1.5;
fontSize = 14;

figure
subplot(1,2,1) %groudtruth 
for t=1:r
    gt(:,t)=B(:,t)/sum(B(:,t)); hold on 
    plot(x,gt(:,t),'color',getprop(colors,t),'linestyle',getprop(linestyle,t),'LineWidth',linew)
    % axis([0 198 0 1])
    set(gca,'xlim',[1 198]);
end
title('Ground-truth','Interpreter','latex','FontSize',fontSize); 
legend('Tree','Water', 'Dirt', 'Road','Interpreter','latex','FontSize',fontSize);  
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize);  
grid on
 
subplot(1,2,2)
for t=1:r
    y(:,t)=W_fgnsr_re(:,t)/sum(W_fgnsr_re(:,t));
    y_spa(:,t)=W_spa_re(:,t)/sum(W_spa_re(:,t));
    hold on
    plot(x,y(:,t),'color',getprop(colors,t),'linestyle',getprop(linestyle,t),'LineWidth',linew)
    % axis([0 198 0 1])
    set(gca,'xlim',[1 198]);
end
title('CSSNMF','Interpreter','latex','FontSize',fontSize); 
legend('Tree','Water', 'Dirt', 'Road','Interpreter','latex','FontSize',fontSize); 
xlabel('Spectral band no.','Interpreter','latex','FontSize',fontSize); 
grid on

%% Compute SSIM - globally

% ssim_CSSNMF=ssim(diag(1./max(H_fgnsr_re,[],2))*H_fgnsr_re,diag(1./max(A,[],2))*A);
% ssim_SSPA=ssim(diag(1./max(H_spa_re,[],2))*H_spa_re,diag(1./max(A,[],2))*A);

ssim_CSSNMF=ssim(H_fgnsr_re,A);
ssim_SSPA=ssim(H_spa_re,A);

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
    term_cssnmf = [term_cssnmf ssim(reshape(H_fgnsr_re(t,:)/max(H_fgnsr_re(t,:)),[100 100]),reshape(A(t,:)/max(A(t,:)),[100 100]))];
end
disp('----------------- CSSNMF --------------------')
term_cssnmf
mean(term_cssnmf)
disp('---------------------------------------------')
term_sspa = [];
for t=1:r
    term_sspa = [term_sspa ssim(reshape(H_spa_re(t,:)/max(H_spa_re(t,:)),[100 100]),reshape(A(t,:)/max(A(t,:)),[100 100]))];
end
disp('----------------- SSPA --------------------')
term_sspa
mean(term_sspa)
disp('---------------------------------------------')
