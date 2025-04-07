addpath(genpath(pwd));
clc
clear
rng(2025) % for reproducibility
%%-------------------------------------------------------------------------
%% Test 2 (see paper)
%%-------------------------------------------------------------------------
n0=50;
m=30;
r=5;

%%number of outliers 
outliers = 1:round(n0/r)+round(1/2*n0/r); iter=10;  

%%metrics for the methods
d=length(outliers); 
err_alg1_av=zeros(iter,d);
err_alg1_med=zeros(iter,d);

%%Initial setting
flag_outliers = 1; 
%%ATTENTION: Choose HERE the kind of normalization
choice_norm = 2;  
  % 0 - Assumption 1 from the paper
  % 1 - prior l-1 normalisation
  % 2 - posterior l-1 normalisation (for l-2, uncomment line 157) 


%%Loop over the number of trials
for j=1:iter

%Case 1: M  with balance (p1=p2=...=pr, where pi is the number of columns of i-th group of H)
%=====================================================================
H0=PB(n0,r,0)';
W=rand(m,r); 
if choice_norm == 1
  H0=H0./sum(H0,1); W=W./sum(W,1);% attention !
end
M0=W*H0; n=n0; 
U=H0; 
U(find(U~=0))=1; % groundtruth of cluster matrix
id_gh=findidx(U');
flag_onmf = 0;   %ATTENTION, set to zero  


%%% Options for Algorithm 2 (post-processing)
options.delta=0.1; % Critical parameter for selecting K in X.
options.type=1;     % Defines the type of spectral clustering algorithm  that should be used. 
options.modeltype=1 - flag_onmf; 
                     % decide to use nnls or alternatingONMF. Usually nnls
                     % for  mix model (1) and alternatingONMF for ONMF
                     % model (0)
options.agregation = 2;
                     % 0 - average 
                     % 1 - median 
                     % 2 - both
options.clustering = 0;
                     % 0 - spectral clustering
                     % 1 - kmeans clustering


for i=1:d
    %%%%%%----------------------------------%%%%%%
    %%%%%% add the noise for Case 1 and Case 2%%%%
    %%%%%%----------------------------------%%%%%%
    if flag_outliers
        numoutliers = outliers(i);
        M = [M0 rand(m,numoutliers)];
    end
    
    %%%%%%----------------------------------%%%%%%
    %%%%%%  normalization of M              %%%%%%
    %%%%%%----------------------------------%%%%%%
    if choice_norm == 2
        % M=M./sqrt(sum(M.^2));
        M=M./sum(M,1);
    end


    %%%%%%----------------------------------%%%%%%
    %%%%%%            Alg.1                 %%%%%%
    %%%%%%----------------------------------%%%%%%
    [X, ~] = fgnsr_alg1(M, r, 'maxiter', 1000); 
    [W_alg1,H_alg1,K_alg1,~] =alg2(M,X,r,options);
    W_alg1_av_re= matchCol(W_alg1{1},W);   % Alg.1 - average
    W_alg1_med_re= matchCol(W_alg1{2},W);  % Alg.1 - median
    err_alg1_av(j,i) = norm(W_alg1_av_re./sum(W_alg1_av_re,1)-W./sum(W,1),'fro')/norm(W./sum(W,1),'fro');
    err_alg1_med(j,i) = norm(W_alg1_med_re./sum(W_alg1_med_re,1)-W./sum(W,1),'fro')/norm(W./sum(W,1),'fro');

end

end
%%-------------------------------------------------------------------------
%% post-processing
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
font_size = 16;
fig(1) = figure;
errorbar(outliers,mean(err_alg1_av,1),std(err_alg1_av,1),'-x','LineWidth',2)
hold on
errorbar(outliers,mean(err_alg1_med,1),std(err_alg1_med,1),'-*','LineWidth',2)
text{1} = 'Alg.1 - Average';
text{2} = 'Alg.1 - Median';
xlabel('number of outliers','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| W^\# - W \|_F}{\| W^\# \|_F}$',"Interpreter",'latex','FontSize',font_size);
legend(text,'Location','northwest','Orientation','horizontal',"Interpreter","latex",'FontSize',font_size)
title('Average plots - $p_t = 10 \quad \forall t=1,...,r=5$',"Interpreter","latex",'FontSize',font_size)
grid on;
% set(gca,'XScale','log');
savefig(fig(1),"Outputs_script/Aver_RelFro_Outliers.fig")


%%%------------------%%%
%%%   Lowest plots   %%%
%%%------------------%%%

%%% Relative Frobenius errors
font_size = 14;
fig(2) = figure;
plot(outliers,min(err_alg1_av),'-x','LineWidth',2)
hold on
plot(outliers,min(err_alg1_med),'-*','LineWidth',2)
text{1} = 'Alg.1 - Average';
text{2} = 'Alg.1 - Median';
xlabel('number of outliers','Interpreter','latex','FontSize',font_size);
ylabel('$\frac{\| W^\# - W \|_F}{\| W^\# \|_F}$',"Interpreter",'latex','FontSize',font_size);
legend(text,'Location','northwest','Orientation','horizontal',"Interpreter","latex",'FontSize',font_size)
title('Best among trials - $p_t = 10 \quad \forall t=1,...,r=5$',"Interpreter","latex",'FontSize',font_size)
grid on;
% set(gca,'XScale','log');
savefig(fig(2),"Outputs_script/Best_RelFro_Outliers.fig")

% 
disp('------------------------------------   Alg.1 ave. |    Alg.1 med.   -----------------')
fprintf('Rel. Frob Error (lower the better) : %2.4e |  %2.4e  \n', mean(err_alg1_av,1),mean(err_alg1_med,1));
disp('----------------------------------------------------------------------------------------------------------')
%     
