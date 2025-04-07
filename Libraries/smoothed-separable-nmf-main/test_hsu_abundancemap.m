% Perform HSU on Urban and generate abundance maps 

% Clean workspace
clear; clc;

% Load util fonctions
addpath('./utils'); 

%% Dataset and parameters

% Urban
displayname = "Urban";
palls = 2400;
psvca = 200;
psspa = 4200;
filename = "data/Urban.mat";
varname = "A";
istr = true;
r = 6;
Li = 307;
Co = 307;
perrow = 6;

% SanDiego
% displayname = "SanDiego";
% psvca = 2600;
% psspa = 2000;
% filename = "data/SanDiego.mat";
% varname = "A";
% istr = true;
% r = 8;
% Li = 400;
% Co = 400;
% perrow = 4;

% Terrain
% displayname = "Terrain";
% psvca = 420;
% psspa = 220;
% filename = "data/Terrain.mat";
% varname = "A";
% istr = true;
% r = 6;
% Li = 500;
% Co = 307;
% perrow = 3;


% Algo parameters
ntrials = 30;


%% Execution
% Load data X 
data = load(filename);
X = data.(varname);
if istr
    X = X';
end

% Filter outliers
ball = [];
for i = 1 : size(X,1)
    [a,b] = sort(X(i,:),'descend');
    ball = [ball, b(1:10)];
end
ball = unique(ball);
X(:,ball) = 0;
fprintf("%s outliers dismissed=%d\n", filename, length(ball));


% Init variables
% besterrALLS = Inf;
besterrVCA = Inf;
besterrSVCA = Inf;
% bestW_ALLS = [];
bestW_VCA = [];
bestW_SVCA = [];
% bestH_ALLS = [];
bestH_VCA = [];
bestH_SVCA = [];

% Run algorithms
% ALLS
% fprintf("ALLS %d runs -", ntrials)
% for t = 1:ntrials
%     fprintf(" %d", t);
%     rng(t);
%     [W,K] = ALLS(X,r,palls);
%     H = NNLS(W,X);
%     err = norm(X-W*H,'fro')/norm(X,'fro')*100;
%     if err < besterrALLS
%         besterrALLS = err;
%         bestH_ALLS = H;
%         bestW_ALLS = W;
%     end 
% end
% fprintf("\n")


% VCA
fprintf("VCA %d runs -", ntrials)
for t = 1:ntrials
    fprintf(" %d", t);
    rng(t);
    [W,K] = SVCA(X,r,1);
    H = NNLS(W,X);
    err = norm(X-W*H,'fro')/norm(X,'fro')*100;
    if err < besterrVCA
        besterrVCA = err;
        bestH_VCA = H;
        bestW_VCA = W;
    end 
end
fprintf("\n")


% SVCA
fprintf("SVCA %d runs -", ntrials)
for t = 1:ntrials
    fprintf(" %d", t);
    rng(t);
    [W,K] = SVCA(X,r,psvca);
    H = NNLS(W,X);
    err = norm(X-W*H,'fro')/norm(X,'fro')*100;
    if err < besterrSVCA
        besterrSVCA = err;
        bestH_SVCA = H;
        bestW_SVCA = W;
    end 
end
fprintf("\n")


% SSPA
fprintf("SPA\n")
[W_SPA,K] = SSPA(X,r,1);
H_SPA = NNLS(W_SPA,X);
errSPA = norm(X-W_SPA*H_SPA,'fro')/norm(X,'fro')*100;


% SSPA
fprintf("SSPA\n")
[W_SSPA,K] = SSPA(X,r,psspa);
H_SSPA = NNLS(W_SSPA,X);
errSSPA =  norm(X-W_SSPA*H_SSPA,'fro')/norm(X,'fro')*100;


%% Display results
%[besterrALLS besterrVCA besterrSVCA errSPA errSSPA]
[besterrVCA besterrSVCA errSPA errSSPA]

% Reorder rows of all H to match SSPA's solution order
% [~, order] = mrsaWs(H_SSPA', bestH_ALLS');
% bestH_ALLS = bestH_ALLS(order,:);
[~, order] = mrsaWs(H_SSPA', bestH_VCA');
bestH_VCA = bestH_VCA(order,:);
[~, order] = mrsaWs(H_SSPA', bestH_SVCA');
bestH_SVCA = bestH_SVCA(order,:);
[~, order] = mrsaWs(H_SSPA', H_SPA');
H_SPA = H_SPA(order,:);

% affichage(bestH_ALLS', perrow, Li, Co, 1);
% saveas(gcf, "./resfigs/alls.fig");
% affichage(bestH_VCA', perrow, Li, Co, 1);
% saveas(gcf, "./resfigs/vca.fig");
% affichage(bestH_SVCA', perrow, Li, Co, 1);
% saveas(gcf, "./resfigs/svca.fig");
% affichage(H_SPA', perrow, Li, Co, 1);
% saveas(gcf, "./resfigs/spa.fig");
% affichage(H_SSPA', perrow, Li, Co, 1);
% saveas(gcf, "./resfigs/sspa.fig");

% Reorder cols of all W
[~, order] = mrsaWs(W_SSPA, bestW_VCA);
bestW_VCA = bestW_VCA(:,order);
[~, order] = mrsaWs(W_SSPA, bestW_SVCA);
bestW_SVCA = bestW_SVCA(:,order);
[~, order] = mrsaWs(W_SSPA, W_SPA);
W_SPA = W_SPA(:,order);

% Save matrices W in text (to plot in tikz)
writematrix(bestW_VCA, strcat(displayname, '_W_vca.txt'), 'Delimiter', 'tab')
writematrix(bestW_SVCA, strcat(displayname, '_W_svca.txt'), 'Delimiter', 'tab')
writematrix(W_SPA, strcat(displayname, '_W_spa.txt'), 'Delimiter', 'tab')
writematrix(W_SSPA, strcat(displayname, '_W_sspa.txt'), 'Delimiter', 'tab')

% Plot spectral signatures
% The pdfcrop lines may only work in Linux
% sigVCA = figure;
% for j = 1:r
%     plot(bestW_VCA(:,j))
%     hold on;
% end
% saveas(sigVCA, "sigVCA.pdf")
% !pdfcrop sigVCA.pdf sigVCA.pdf
% hold off;
% 
% sigSVCA = figure;
% for j = 1:r
%     plot(bestW_SVCA(:,j))
%     hold on;
% end
% saveas(sigSVCA, "sigSVCA.pdf")
% !pdfcrop sigSVCA.pdf sigSVCA.pdf
% hold off;
% 
% sigSPA = figure;
% for j = 1:r
%     plot(W_SPA(:,j))
%     hold on;
% end
% saveas(sigSPA, "sigSPA.pdf")
% !pdfcrop sigSPA.pdf sigSPA.pdf
% hold off;
% 
% sigSSPA = figure;
% for j = 1:r
%     plot(W_SSPA(:,j))
%     hold on;
% end
% saveas(sigSSPA, "sigSSPA.pdf")
% !pdfcrop sigSSPA.pdf sigSSPA.pdf
% hold off;
% 

