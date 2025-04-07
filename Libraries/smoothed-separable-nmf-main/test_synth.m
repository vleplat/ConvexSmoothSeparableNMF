% Test on synthetic data
clear all; clc; 

% Parameters
r = 10;
n = 1000; 
purity = 0.05; 
noislevel = logspace(-2,0,20); 
p = [1 20 50]; 
%p = [1 50 100 200]; 
ntrials = 30;
ndata = 10;

% Add utils and data 
addpath('./data'); 
addpath('./utils'); 

% load Wsynth; 
% [m,r] = size(W); 
% Wt = W;

load USGS_Library;
data = datalib(:,5:end); % first 4 cols of usgs are not materials
[m,nbcol] = size(data);

% Preallocate output arrays
err_alls = zeros(length(p), length(noislevel), ntrials, ndata);
err_svca = zeros(length(p), length(noislevel), ntrials, ndata);
err_sspa = zeros(length(p), length(noislevel), ndata);
legendstrings_alls = strings(1,length(p));
legendstrings_svca = strings(1,length(p));
legendstrings_sspa = strings(1,length(p));

% Perfom experiments
disp('*********************************')
for i = 1 : length(noislevel)
    for j = 1 : length(p)
        fprintf("%d %d\n", i, p(j));
        for nd = 1 : ndata
        
            % Generate W
            selcol = randperm(nbcol, r);
            W = data(:,selcol);
            Wt = W;

            % Generate H and X
            Ht = [eye(r) sample_dirichlet(ones(r,1)*purity, n-r)'];
            X = Wt*Ht; 
            Noise = randn(m,n);  
            Xn = X + noislevel(i) * Noise/norm(Noise,'fro') * norm(X,'fro');

            % Run ALLS
            for t = 1 : ntrials
                rng(t); % Seed for reproductible random
                [W,K] = ALLS(Xn, r, p(j));
                err_alls(j,i,t,nd) = mrsaWs(Wt, W);
            end
            legendstrings_alls(j) = sprintf("ALLS(%d)", p(j));

            % Run SVCA
            for t = 1 : ntrials
                rng(t); % Seed for reproductible random
                [W,K] = SVCA(Xn, r, p(j));
                err_svca(j,i,t,nd) = mrsaWs(Wt, W);
            end
            legendstrings_svca(j) = sprintf("SVCA(%d)", p(j));

            % Run SSPA (only once because it is determinist)
            [W,K] = SSPA(Xn, r, p(j));
            err_sspa(j,i,nd) = mrsaWs(Wt, W);
            legendstrings_sspa(j) = sprintf("SSPA(%d)", p(j));
        end
    end
end

% Median of the error per parameter set over all trials
mederr_alls = median(err_alls, [3 4]);
mederr_svca = median(err_svca, [3 4]);
mederr_sspa = median(err_sspa, 3);

% Build matrix for convenient export to text (to use in latex/pgfplots)
% data_alls = [noislevel' mederr_alls']
% data_svca = [noislevel' mederr_svca']
% data_sspa = [noislevel' err_sspa']
results = [noislevel' mederr_alls' mederr_svca' mederr_sspa'];
results

% Plots
% figure; 
% semilogx(noislevel,mederr_svca','o--'); hold on;
% semilogx(noislevel,err_sspa','-o'); 
% legend([legendstrings_svca legendstrings_sspa],'Interpreter','latex');
% xlabel('$\epsilon$','Interpreter','latex'); 
% ylabel('MRSA','Interpreter','latex');
% hold off;
