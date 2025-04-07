function [X, out] = fminBFGS(X0,fun,opts,varargin)
%-------------------------------------------------------------------------
% This code is designed to solve
%  min 0.5*\|X - A\|_F^2, 
%  s.t. X>=0, Xe = e, X'e = e.
%
%  Input:
%     A  n by n matrix
%     opts -- option structure with fields
%         record    0, no print-out; else with print-out
%         maxiter   max number of iterations
%         gtol      stop control for the gradient norm
%
%   Output:
%           X      doubly stochastic matrix solution
%           msg    convergence state
%           normg   norm of dual gradient

%  ----------------------------------------------------------------------
%% set the model
check_param;

%% -------------------           copy parameters  -----------------------
gtol = opts.gtol;
maxiter = opts.maxiter;
record = opts.record;
%%  --------------------   copying parameters ---------------------

%  ------         prepare for iteration
n = length(X0)/2;
[X, out] = sBFGS(X0);

%%   %%%%  nested funciton  %%%%%
    function check_param
        if ~isfield(opts,'record')
            opts.record = 0;
        end
        if isfield(opts,'maxiter')
            if opts.maxiter <=0
                error('opts.maxiter should be in [1,Inf)');
            end
        else
            opts.maxiter = 2000;
        end
        if isfield(opts,'gtol')
            if opts.gtol <0
                error('opts.gtol should be in [0,inf');
            end
        else
            opts.gtol = 1e-6;
        end
    end

%%  %%%%%  Our s-BFGS Procedure  %%%%%
    function [X,output] = sBFGS(X)
        
        [G, H, Dst] = feval(fun,X,varargin{:});
        d = -G./H;
        normG= norm(G,'fro');
        
        Tol = gtol;
        for iter = 1: maxiter
            
            % test the stopping condition
            if normG <= Tol
                msg = 'convergence';
                break;
            end
            
            % check the angle between -d and gradient
            cos_tk = -d'*G/norm(d)/normG;
            if(cos_tk < 1/n)
                d = -G.*H;
            end
            
            Xold = X; dold = d; Gold = G;
            
            %%  compute the step size and the next point
            d_a = d(1:n);
            d_b = d(n+1:end);
            
            dMat = -bsxfun(@plus, d_a, d_b');
            Inx = Dst.Inx;
            dM = dMat.*Inx;
            dMvec = dM(:);
            
            g_step = G'*d;   % projG'*dMvec + sum(d);
            H_step = dMvec'*dMvec;
            tau = -g_step/H_step;  % find the step size based on Newton's method
            
            X = Xold + tau*dold;
            [G, H, Dst] = feval(fun,X,varargin{:});
            H = 1./H;
            
            normG = norm(G,'fro');
            normGArray(iter) = normG;
            if(record)
                fprintf('iter:%d, ||G||:%.12f, stepsize: %.4f \n',...
                    iter, normG, tau);
            end
            
            sk = X - Xold;
            yk = G - Gold;
            
            sty = sk'*yk;
            if(sty<=0)
                disp('The curvature condition violated.');
            end
            rouk = 1/sty;
            Vk_g = -G + rouk*(sk'*G).*yk;
            HkVkg = H .* Vk_g;
            VHVg = HkVkg - rouk * (yk'*HkVkg) .* sk;
            d = VHVg - rouk * (sk'*G) .* sk;
        end
        
        if iter >=maxiter
            msg = 'exceed max iteration';
        end
        
        output.msg = msg;
        output.normG = normG;
        output.iter = iter-1;
        output.normGArray = normGArray;
    end

end


