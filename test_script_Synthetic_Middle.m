addpath(genpath(pwd));
clc
clear
rng(2025)
%%-------------------------------------------------------------------------
%% Test 3 - additional test - Middle points and adversarial noise
% We generate the noisy matrices exactly as in the Test 1, 
% $$
% \max(0,M_0+N)\quad M_0=W[H_0,H_1]
% $$
% $W$ is drawn from a uniform distribution over the interval $[0,1]$ and $H_0$ is in the form of (\ref{H}), except the $\binom{r}{2} = 10$ columns $H_1$
% contain all possible combinations of two non-zero entries equal to $0.5$ at different positions.
% Hence, the columns of $WH_1$ are all the middle points of the columns of $W$. 
% 
% No noise is added to the first $n_0$ columns of $M_0$, that is, $N(:,1:n_0)=0$, while we set $N(:,n_0+1:n)$ equal to
%       $$
%        M_0(:,n_0+1:n) -
%       \bar{w} e^T,
%       $$
%       where
%       $e$ is the vector of all ones of appropriate dimension, and $\bar{w}$ is the average of the columns of $WH_0$, that is, $\bar{w} = \frac{1}{n_0} WH_0e$.
% The noise matrix $N$ is normalized so that $||N||_F = \epsilon ||M_0||_F$.
% 
% This example is inspired by the so-called middle point experiment from~\cite{gillis2014fast}. 
% Intuitively, we are moving the data points towards the outside of the set spanned by $WH_0$. All the algorithms are implemented in the normalized matrix $M$.
%%-------------------------------------------------------------------------
n0=50;
m=30;
r=5;

%%noise level 
% epsilon = logspace(-5,-0.05,7); iter=20;  
epsilon = logspace(-3,-0.05,4); iter=20;

%%metrics for the methods
d=length(epsilon); 
res_fgnsr=zeros(iter,d); err_fgnsr=zeros(iter,d); distW_fgnsr=zeros(iter,d); 
res_spa=zeros(iter,d);  err_spa=zeros(iter,d); distW_spa=zeros(iter,d); 
res_alg1=zeros(iter,d); err_alg1=zeros(iter,d); distW_alg1=zeros(iter,d); 
res_sspa=zeros(iter,d);  err_sspa=zeros(iter,d); distW_sspa=zeros(iter,d); 

%%Initial setting
flag_noise = 1; 
%%ATTENTION: Choose HERE the kind of normalization
choice_norm = 2;  
  % 0 - Assumption 1 from the paper - H is column stochastic (not relevant
  % anymore)
  % 1 - prior l-1 normalisation  - both W and H are column-stochastic, then
  % noise is added -> Better to use k-means for post-processing
  % 2 - posterior l-1 normalisation (for l-2, uncomment line 157) 


%%Loop over the number of trials
for j=1:iter

%%% M =W[H0 H1], H0 is ONMF and H1 contain all possible combinations of two non-zero entries equal to 0.5 at
% different positions.
H0=PB(n0,r,1)';
n1=10; n=n0+n1;
v = 1:1:5;
C = nchoosek(v,2);
H1=zeros(5,10);
for i=1:10
    H1(C(i,:)',i)=1/2;
end  
H=[H0,H1]; 
W=rand(m,r); 
if choice_norm == 1
  H=H./sum(H,1); W=W./sum(W,1);% attention !
elseif choice_norm==0
  H=H./sum(H,1);  
end
M0=W*H; flag_onmf = 0;
U=H./sum(H); 
M = M0;

%%% Display in the 3d case
if m==3
    Mdis = M; Wdis = W;
    figure
    plot3(Mdis(1,:),Mdis(2,:),Mdis(3,:),'o')
    hold on
    plot3(Wdis(1,:),Wdis(2,:),Wdis(3,:),'s')
    axis equal
    grid on
    title('Columns of M (blue) and W (red) - noiseless','Interpreter','latex','FontSize',14)
    xlabel('x component','Interpreter','latex','FontSize',14)
    ylabel('y component','Interpreter','latex','FontSize',14)
    zlabel('z component','Interpreter','latex','FontSize',14)
    keyboard
end

%%% Options for Algorithm 2 (post-processing)
options.delta=0.95; % Critical parameter for selecting K in X.
options.type=1;     % Defines the type of spectral clustering algorithm  that should be used. 
options.modeltype=1 - flag_onmf; 
                     % decide to use nnls or alternatingONMF. Usually nnls
                     % for  mix model (1) and alternatingONMF for ONMF
                     % model (0)
options.agregation = 1;
                     % 0 - average 
                     % 1 - median 
options.clustering = 0;
                     % 0 - spectral clustering
                     % 1 - kmeans clustering

%%% for SSPA -  approximation of the number of proximal latent points
nplp = 0;
for f=1:r
    nplp = nplp + length(find(H0(f,:)~=0));
end
nplp = round(nplp/r); % 1/r \sum_t p_t
% nplp = 1;           % min_t p_t

for i=1:d
    %%%%%%----------------------------------%%%%%%
    %%%%%%          Add the noise           %%%%%%
    %%%%%%----------------------------------%%%%%%
    if flag_noise
        Noise=randn(m,n); 
        Noise(:,1:n0) = 0;
        wbar = mean(W*H0,2);
        Noise(:,n0+1:end) = M0(:,n0+1:end)-wbar*ones(1,n1);
        Noise=epsilon(i)*(Noise/norm(Noise,'fro'))*norm(M0,'fro'); 
        M=max(M0+Noise,0); 
        if m==3
            Mdis = M; Wdis = W;
            figure
            plot3(Mdis(1,:),Mdis(2,:),Mdis(3,:),'o')
            hold on
            plot3(Wdis(1,:),Wdis(2,:),Wdis(3,:),'s')
            axis equal
            grid on
            title('Columns of M (blue) and W (red) - with noise','Interpreter','latex','FontSize',14)
            xlabel('x component','Interpreter','latex','FontSize',14)
            ylabel('y component','Interpreter','latex','FontSize',14)
            zlabel('z component','Interpreter','latex','FontSize',14)
            keyboard
        end

    end
    
    %%%%%%----------------------------------%%%%%%
    %%%%%%   Posterior normalization of M   %%%%%%
    %%%%%%----------------------------------%%%%%%
    if choice_norm == 2
        % M=M./sqrt(sum(M.^2));
        M=M./sum(M,1);
    end
    
    %%
    %%%%%%----------------------------------%%%%%%
    %%%%%%               Alg.1              %%%%%%
    %%%%%%----------------------------------%%%%%%
    [X_alg1, K_alg1_1] = fgnsr_alg1(M, r, 'maxiter', 1000); %5000, 50, 1000 (2.2)
    % [X_alg1, K_alg1_1] = fgnsr_alg1(M, r, 'maxiter', 1500, 'mu',0.5);
    [W_alg1,H_alg1,K_alg1,Walg1] =alg2(M,X_alg1,r,options);
    %%%%% compute the residual %%%%%
    res_alg1(j,i)=norm(M-W_alg1*H_alg1,'fro')./norm(M,'fro');
    H_alg1=matchCol(H_alg1',U',W')';

    %%%%% compute the clustering error %%%%%
    [err_alg1(j,i),~] = Compare_clustering(H_alg1(:,1:n0)',U(:,1:n0)',0,~flag_onmf); %%% note the 4th parameter: 0 for ONMF, 1 for mix-model.%%%
    
    %%%%% compute the relative distance between W estimated and the ground
    %%%%% truth W
    W_alg1_av_re= matchCol(W_alg1,W);
    distW_alg1(j,i)=norm(W_alg1_av_re./sum(W_alg1_av_re,1)-W./sum(W,1),'fro')/norm(W./sum(W,1),'fro');
  

    %%
    %%%%%%----------------------------------%%%%%%
    %%%%%%               FGM                %%%%%%
    %%%%%%----------------------------------%%%%%%
    % [X_fgnsr, K_fgnsr] = fgnsr(M, r, 'maxiter', 1000,'mu',0); 
    [X_fgnsr, K_fgnsr] = fgnsr(M, r, 'maxiter', 1000); 
    W_fgnsr = M(:,K_fgnsr) ;
    H_fgnsr = nnlsHALSupdt_new(W_fgnsr'*M,W_fgnsr,[],1000);   

    %%%%% compute the residual %%%%%
    res_fgnsr(j,i)=norm(M-W_fgnsr*H_fgnsr,'fro')./norm(M,'fro');
    %%%%% compute the clustering error %%%%% 
    H_fgnsr=matchCol(H_fgnsr',U',W')';
    %%%%% compute the clustering error %%%%%
    [err_fgnsr(j,i),~] = Compare_clustering(H_fgnsr(:,1:n0)',U(:,1:n0)',0,~flag_onmf); %%% note the 4th parameter: 0 for ONMF, 1 for mix-model.%%%
     
    %%%%% compute the relative distance between W estimated and the ground
    %%%%% truth W
    W_fgnsr_re= matchCol(W_fgnsr,W);
    distW_fgnsr(j,i)=norm(W_fgnsr_re./sum(W_fgnsr_re,1)-W./sum(W,1),'fro')/norm(W./sum(W,1),'fro');
  
    %% 
    %%%%%%----------------------------------%%%%%%
    %%%%%%               SPA                %%%%%%
    %%%%%%----------------------------------%%%%%%
    %%% Call of SPA
    K_spa=FastSepNMF(M, r, 0);
    W_spa=M(:,K_spa);
    H_spa=nnlsHALSupdt_new(W_spa'*M,W_spa,[],1000);   
    
    %%%%% compute the residual %%%%%
    res_spa(j,i)=norm(M-W_spa*H_spa,'fro')./norm(M,'fro');
    %%%%% compute the clustering error %%%%%
    H_spa=matchCol(H_spa',U',W')';
    [err_spa(j,i),~] = Compare_clustering(H_spa(:,1:n0)',U(:,1:n0)',0,~flag_onmf);  %%% note the 4th parameter: 0 for ONMF, 1 for mix-model.%%%

    %%%%% compute the relative distance between W estimated and the ground
    %%%%% truth W
    W_spa_re= matchCol(W_spa,W);
    distW_spa(j,i)=norm(W_spa_re./sum(W_spa_re,1)-W./sum(W,1),'fro')/norm(W./sum(W,1),'fro');
  
   
    %% 
    %%%%%%----------------------------------%%%%%%
    %%%%%%              SSPA                %%%%%%
    %%%%%%----------------------------------%%%%%%
    %%% Call of SSPA
    [W_sspa,K_sspa] = SSPA(M, r, nplp);
    H_sspa=nnlsHALSupdt_new(W_sspa'*M,W_sspa,[],1000);   

    %%%%% compute the residual %%%%%
    res_sspa(j,i)=norm(M-W_sspa*H_sspa,'fro')./norm(M,'fro');

    %%%%% compute the clustering error %%%%%
    H_sspa=matchCol(H_sspa',U',W')';
    [err_sspa(j,i),~] = Compare_clustering(H_sspa(:,1:n0)',U(:,1:n0)',0,~flag_onmf);  %%% note the 4th parameter: 0 for ONMF, 1 for mix-model.%%%

    %%%%% compute the relative distance between W estimated and the ground
    %%%%% truth W
    W_sspa_re= matchCol(W_sspa,W);
    distW_sspa(j,i)=norm(W_sspa_re./sum(W_sspa_re,1)-W./sum(W,1),'fro')/norm(W./sum(W,1),'fro');
  

    %%
    %%%%%%----------------------------------%%%%%%
    %%%%%%     Display some results         %%%%%%
    %%%%%%----------------------------------%%%%%%
    disp('----------------------------------------------------------------------------------------------------------')
    disp('------------------------------------   Alg.1 |    FGNSR     |    SSPA    -----------------')
    fprintf('Rel. Frob Error (lower the better) : %2.4e |  %2.4e |  %2.4e  \n', res_alg1(j,i),res_fgnsr(j,i),res_sspa(j,i));
    fprintf('Clustering Error (lower the better): %2.4e |  %2.4e |  %2.4e  \n', err_alg1(j,i),err_fgnsr(j,i),err_sspa(j,i));
    fprintf('Rel. d(W^#,W)     : %2.4e |  %2.4e |  %2.4e \n', distW_alg1(j,i),distW_fgnsr(j,i),distW_sspa(j,i));
    disp('----------------------------------------------------------------------------------------------------------')
    

end

end
%%-------------------------------------------------------------------------
%% Post-processing
%%-------------------------------------------------------------------------
yourFolder = 'Outputs_script';
if not(isfolder(yourFolder))
    mkdir(yourFolder)
end
close all
%%%------------------%%%
%%%   Average plots  %%%
%%%------------------%%%
%%% Relative Frobenius errors
font_size = 14;
fig(1) = figure;
errorbar(epsilon,mean(res_fgnsr,1),std(res_fgnsr,1),'-x','LineWidth',2)
hold on
errorbar(epsilon,mean(res_spa,1),std(res_spa,1),'-*','LineWidth',2)
hold on
errorbar(epsilon,mean(res_alg1,1),std(res_alg1,1),'-','LineWidth',2)
hold on
errorbar(epsilon,mean(res_sspa,1),std(res_sspa,1),'-.','LineWidth',2)
text{1} = 'FGM';
text{2} = 'SPA';
text{3} = 'Alg.1';
text{4} = 'SSPA';
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| M - WH \|_F}{\| M \|_F}$',"Interpreter",'latex','FontSize',font_size);
legend(text,'Location','northwest','Orientation','horizontal',"Interpreter","latex",'FontSize',font_size)
title('Average plots - Relative Frobenius Errors',"Interpreter","latex",'FontSize',font_size)
grid on;
set(gca,'XScale','log');
savefig(fig(1),"Outputs_script/Aver_RelFro.fig")
%%% Accuracy 
font_size = 14;
fig(2) = figure;
errorbar(epsilon,mean(1-err_fgnsr,1),std(1-err_fgnsr,1),'-x','LineWidth',2)
hold on
errorbar(epsilon,mean(1-err_spa,1),std(1-err_spa,1),'-*','LineWidth',2)
hold on
errorbar(epsilon,mean(1-err_alg1,1),std(1-err_alg1,1),'-','LineWidth',2)
hold on
errorbar(epsilon,mean(1-err_sspa,1),std(1-err_sspa,1),'-.','LineWidth',2)
ylim([0 1])
text{1} = 'FGM';
text{2} = 'SPA';
text{3} = 'Alg.1';
text{4} = 'SSPA';
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('Accuracy',"Interpreter",'latex','FontSize',font_size);
legend(text,'Location','northwest','Orientation','horizontal',"Interpreter","latex",'FontSize',font_size)
title('Average plots - Accuracy',"Interpreter","latex",'FontSize',font_size)
grid on;
set(gca,'XScale','log');
savefig(fig(2),"Outputs_script/Aver_Acc.fig")

%%% Relative Distance \|W^# - W\|_F/ \|W^#\|_F
fig(3) = figure;
semilogx(epsilon,mean(distW_fgnsr,1),'-x','LineWidth',2)
hold on
semilogx(epsilon,mean(distW_spa,1),'-*','LineWidth',2)
hold on
semilogx(epsilon,mean(distW_alg1,1),'-','LineWidth',2)
hold on
semilogx(epsilon,mean(distW_sspa,1),'-.','LineWidth',2)
ylim([0 1])
text{1} = 'FGM';
text{2} = 'SPA';
text{3} = 'Alg.1';
text{4} = 'SSPA';
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| W^\# - W \|_F}{\| W^\# \|_F}$',"Interpreter",'latex','FontSize',font_size);
legend(text,'Location','northwest','Orientation','horizontal',"Interpreter","latex",'FontSize',font_size)
title('Average plots - Relative distance w.r.t. $W^\#$',"Interpreter","latex",'FontSize',font_size)
grid on;
savefig(fig(3),"Outputs_script/Aver_distW.fig")

%%%------------------%%%
%%%   Min plots      %%%
%%%------------------%%%

%%% Relative Frobenius errors
font_size = 14;
fig(4) = figure;
semilogx(epsilon,min(res_fgnsr),'-x','LineWidth',2)
hold on
semilogx(epsilon,min(res_spa),'-*','LineWidth',2)
hold on
semilogx(epsilon,min(res_alg1),'-','LineWidth',2)
hold on
semilogx(epsilon,min(res_sspa),'-.','LineWidth',2)
text{1} = 'FGM';
text{2} = 'SPA';
text{3} = 'Alg.1';
text{4} = 'SSPA';
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| M - WH \|_F}{\| M \|_F}$',"Interpreter",'latex','FontSize',font_size);
legend(text,'Location','northwest','Orientation','horizontal',"Interpreter","latex",'FontSize',font_size)
title('Best among trials - Relative Frobenius Errors',"Interpreter","latex",'FontSize',font_size)
grid on;
set(gca,'XScale','log');
savefig(fig(4),"Outputs_script/Best_RelFro.fig")

%%% Accuracy 
font_size = 14;
fig(5) = figure;
semilogx(epsilon,max(1-err_fgnsr),'-x','LineWidth',2)
hold on
semilogx(epsilon,max(1-err_spa),'-*','LineWidth',2)
hold on
semilogx(epsilon,max(1-err_alg1),'-','LineWidth',2)
hold on
semilogx(epsilon,max(1-err_sspa),'-.','LineWidth',2)
ylim([0 1])
text{1} = 'FGM';
text{2} = 'SPA';
text{3} = 'Alg.1';
text{4} = 'SSPA';
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('Accuracy',"Interpreter",'latex','FontSize',font_size);
legend(text,'Location','northwest','Orientation','horizontal',"Interpreter","latex",'FontSize',font_size)
title('Best among trials - Accuracy',"Interpreter","latex",'FontSize',font_size)
grid on;
set(gca,'XScale','log');
savefig(fig(5),"Outputs_script/Best_Acc.fig")

%%% Relative Distance \|W^# - W\|_F/ \|W^#\|_F
font_size = 14;
fig(6)=figure;
semilogx(epsilon,min(distW_fgnsr),'-x','LineWidth',2)
hold on
semilogx(epsilon,min(distW_spa),'-*','LineWidth',2)
hold on
semilogx(epsilon,min(distW_alg1),'-','LineWidth',2)
hold on
semilogx(epsilon,min(distW_sspa),'-.','LineWidth',2)
ylim([0 1])
text{1} = 'FGM';
text{2} = 'SPA';
text{3} = 'Alg.1';
text{4} = 'SSPA';
xlabel('level of noise - $\epsilon$','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| W^\# - W \|_F}{\| W^\# \|_F}$',"Interpreter",'latex','FontSize',font_size);
legend(text,'Location','northwest','Orientation','horizontal',"Interpreter","latex",'FontSize',font_size)
title('Best among trials - Relative distance w.r.t. $W^\#$',"Interpreter","latex",'FontSize',font_size)
grid on;
%set(gca,'XScale','log');
savefig(fig(6),"Outputs_script/Best_distW.fig")


