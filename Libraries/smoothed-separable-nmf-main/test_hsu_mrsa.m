% Test SVCA and SSPA on hyperspectral unmixing

% Clean workspace
clear; clc;

% Load util fonctions
addpath('./utils'); 

% Run on several datasets
% runtesthsu("data/Cuprite.mat", "x", false, 12, [1 100 1000 2000], 20);
% runtesthsu("data/SanDiego.mat", "A", true, 8, [1 100 1000 2000 5000], 20);
% runtesthsu("data/Urban.mat", "A", true, 6, [1 100 1000 2000 5000], 20);
% runtesthsu("data/Terrain.mat", "A", true, 6, [1 100 1000 2000 5000], 20);

% Run measring MRSA when GT is available
runtesthsu("data/Cuprite.mat", "x", false, 12, [1 100 1000 2000], 20, ...
           "data/Cuprite_Ref.mat", "Wgt", false);
runtesthsu("data/Urban.mat", "A", true, 6, [1 100 1000 2000 5000], 20, ...
           "data/Urban_Ref.mat", "References", true);


% Generic test function
function runtesthsu(filename, varname, istr, r, p, ntrials, ...
                    gtfilename, gtvarname, gtistr)
    % If groundtruth is provided, compute MRSA instead of error
    compmrsa = false;
    if nargin > 6
        compmrsa = true;
    end

    % Load data X 
    data = load(filename);
    X = data.(varname);
    if istr
        X = X';
    end
    
    % Load groundtruth Wgt if needed
    if compmrsa
        data = load(gtfilename);
        Wgt = data.(gtvarname);
        if gtistr
            Wgt = Wgt';
        end
    end
        
    % Run algorithms
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
            if compmrsa
                taberrALLS(t) = mrsaWs(Wgt, W);
            else
                H = NNLS(W,X);
                taberrALLS(t) =  norm(X-W*H,'fro')/norm(X,'fro')*100; 
            end
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
            if compmrsa
                taberrSVCA(t) = mrsaWs(Wgt, W);
            else
                H = NNLS(W,X);
                taberrSVCA(t) =  norm(X-W*H,'fro')/norm(X,'fro')*100; 
            end
        end
        fprintf("\n")
        
        % SSPA
        tic;
        [W,K] = SSPA(X,r,p(i));
        timeSSPA = toc;
        if compmrsa
            errSSPA = mrsaWs(Wgt, W);
        else
            H = NNLS(W,X);
            errSSPA =  norm(X-W*H,'fro')/norm(X,'fro')*100; 
        end
        
        % Print results
        fprintf("Algo\ttime\tmin err\tmed err\tmax err\n")
        fprintf("ALLS\t%2.2f\t%2.2f\t%2.2f\t%2.2f\n", median(tabtimeALLS), min(taberrALLS), median(taberrALLS), max(taberrALLS));
        fprintf("SVCA\t%2.2f\t%2.2f\t%2.2f\t%2.2f\n", median(tabtimeSVCA), min(taberrSVCA), median(taberrSVCA), max(taberrSVCA));
        fprintf("SSPA\t%2.2f\t%2.2f\n", timeSSPA, errSSPA)
    end

end