% Test SVCA and SSPA on hyperspectral unmixing

% Clean workspace
clear; clc;

% Load util fonctions
addpath('./utils'); 

% Run on several datasets
resCu = runtesthsu("data/Cuprite.mat", "x", false, 12, [1 100 1000 2000], 30);
resSD = runtesthsu("data/SanDiego.mat", "A", true, 8, [1 100 1000 2000 5000], 30);
resUr = runtesthsu("data/Urban.mat", "A", true, 6, [1 100 1000 2000 5000], 30);
resTe = runtesthsu("data/Terrain.mat", "A", true, 6, [1 100 1000 2000 5000], 30);


% Generic test function
function tabresults = runtesthsu(filename, varname, istr, r, p, ntrials)

    % Load data X 
    data = load(filename);
    X = data.(varname);
    if istr
        X = X';
    end
    
    % Preprocess outliers
    ball = [];  
    for i = 1 : size(X,1) 
        [a,b] = sort(X(i,:),'descend'); 
        ball = [ball, b(1:10)]; 
    end
    ball = unique(ball); 
    X(:,ball) = 0;
    fprintf("******************** %s outliers dismissed=%d\n", filename, length(ball));
  
    % Run algorithms
    tabresults = zeros(length(p)*3, 4);
    for i = 1:length(p)
        fprintf("******************** %s p=%d\n", filename, p(i));
        
        % ALLS
        taberrALLS = zeros(ntrials,1);
        tabtimeALLS = zeros(ntrials,1);
        fprintf("ALLS %d runs -", ntrials)
        for t = 1:ntrials
            fprintf(" %d", t);
            rng(t);
            tic;
            [W,K] = ALLS(X,r,p(i));
            tabtimeALLS(t) = toc;
            H = NNLS(W,X);
            taberrALLS(t) =  norm(X-W*H,'fro')/norm(X,'fro')*100; 
        end
        fprintf("\n")
        
        
        % SVCA
        taberrSVCA = zeros(ntrials,1);
        tabtimeSVCA = zeros(ntrials,1);
        fprintf("SVCA %d runs -", ntrials)
        for t = 1:ntrials
            fprintf(" %d", t);
            rng(t);
            tic;
            [W,K] = SVCA(X,r,p(i));
            tabtimeSVCA(t) = toc;
            H = NNLS(W,X);
            taberrSVCA(t) =  norm(X-W*H,'fro')/norm(X,'fro')*100; 
        end
        fprintf("\n")
        
        % SSPA
        tic;
        [W,K] = SSPA(X,r,p(i));
        timeSSPA = toc;
        H = NNLS(W,X);
        errSSPA =  norm(X-W*H,'fro')/norm(X,'fro')*100; 
        
        % Print results
        fprintf("Algo\ttime\tmin err\tmed err +- std\tmax err\n")
        fprintf("ALLS\t%2.2f\t%2.2f\t%2.2f +- %2.2f\t%2.2f\n", median(tabtimeALLS), min(taberrALLS), median(taberrALLS), std(taberrALLS), max(taberrALLS));
        fprintf("SVCA\t%2.2f\t%2.2f\t%2.2f +- %2.2f\t%2.2f\n", median(tabtimeSVCA), min(taberrSVCA), median(taberrSVCA), std(taberrSVCA), max(taberrSVCA));
        fprintf("SSPA\t%2.2f\t%2.2f\n", timeSSPA, errSSPA)
        
        tabresults(i,:) = [min(taberrALLS) median(taberrALLS) std(taberrALLS) max(taberrALLS)];
        tabresults(i+length(p),:) = [min(taberrSVCA) median(taberrSVCA) std(taberrSVCA) max(taberrSVCA)];
        tabresults(i+2*length(p),:) = [errSSPA 0 0 0];
    end

end