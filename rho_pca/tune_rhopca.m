function out = tune_rhopca(X, varargin)
%==========================================================================
% This function tunes rho-PCA by selecting the optimal lambda_v, lambda_w,
% alpha_w, and alpha_t for each k = 1, ..., K components. Parameters are
% selected by a nested BIC routine.
%
% INPUT:
%   1. 'X': An 4-dimensional tensor
%   2. 'K' (default 1): the number of components, K > 0
%   3. 'candidate_lamV' (default 0): Vector of positive sparsity penalty
%   for electrode dimension (vector if K > 1)
%   4. 'candidate_lamW' (default 0): Vectir if positive lasso parameter
%   associated with frequency dimension (vector if K > 1)
%   5. 'candidate_alphaW' (default 0): Vector of non-negative multiplier of
%   omegaW (vector if K > 1)
%   6. 'candidate_alphaT' (default 0): Non-negative multplier of omegaT
%   (vector if K > 1)
%   7. 'warmstart' (default true): Optional, if true, use
%       CP-decomposition to initialize rhoPCA
%   8. 'maxIterations' (default 500): Optional, maximum number of
%       alternating regressions steps
%   9. 'threshold' (default 1e-6): Optional, convergence for change in
%       objective value
%   10. 'verbose' (default true): Optional, boolean value indicating to
%       print all messages in code
%   11. 'par_on' (default false): Optional, boolean value indicating to use
%   Parallel Toolbox
%   12. 'n_workers' (default 1): Optional, number of workers to use
%
% OUTPUT:
%   1. 'out': Metadata, contains parameters and convergence information
%
% Dependencies: Tensor Toolbox package, soft_thr.m
%==========================================================================

%% Load and verify parameters
% Specify default parameters and their expected values
default_K = 1;
default_maxit = 500;
default_threshold = 1e-4;
default_lam = 0;
default_alpha = 0;
default_verbose = true;
default_warmstart = true;
default_paron = false;
default_nworkers = 1;
% Parse input parameters
params = inputParser;
params.CaseSensitive = true;
addRequired(params, 'X', @(ii) isa(ii,'double')||isa(ii,'tensor'));
addParameter(params, 'K', default_K, @(ii) (isnumeric(ii)&&(ii > 0)))
addParameter(params, 'candidate_lamV', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'candidate_lamW', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'candidate_alphaW', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'candidate_alphaT', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'maxIterations', default_maxit, @(ii) isnumeric(ii)&&(ii>1));
addParameter(params, 'threshold', default_threshold, @(ii) isnumeric(ii)&&(ii>0));
addParameter(params, 'verbose', default_verbose, @islogical);
addParameter(params, 'warmstart', default_warmstart, @islogical);
addParameter(params, 'par_on', default_paron, @islogical);
addParameter(params, 'n_workers', default_nworkers, @(ii) (isnumeric(ii)&&(ii >=1)));
parse(params, X, varargin{:});
K = params.Results.K;
lamV = params.Results.candidate_lamV;
lamW = params.Results.candidate_lamW;
alphaW = params.Results.candidate_alphaW;
alphaT = params.Results.candidate_alphaT;
maxit = params.Results.maxIterations;
verbose = params.Results.verbose;
warmstart = params.Results.warmstart;
thr = params.Results.threshold;
par_on = params.Results.par_on;
n_workers = params.Results.n_workers;
% Check dimensionality of covariates X
X = squeeze(X);         % Remove singletion dimensions, if present
nD = length(size(X));
if nD ~= 4; error('Data must be a 4-dimensional tensor'); end
X = double(X);
% Convert X to tensor class
if ~isa(X, 'tensor'); X = tensor(X); end

%% Set up parallel environment
% Set up parallel options
par_opts = 0;
if par_on
    myCluster = parcluster('local');
    if getenv('ENVIRONMENT')
        myCluster.JobStorageLocation = getenv('TMPDIR');
    end
    pool = parpool(myCluster, n_workers);
    par_opts = parforOptions(pool);
end

%% Set up nested BIC
% Sort candidate parameters in ascending order
lamV = sort(lamV);
lamW = sort(lamW);
alphaW = sort(alphaW);
alphaT = sort(alphaT);
% Load dimensions
nDims = size(X); n = nDims(1); p = nDims(2); q = nDims(3); r = nDims(4);
% Set both omegaW and omegaT to be 2nd differencing matrices
omegaT = gallery('tridiag', r); 
omegaW = gallery('tridiag', q);
% Store pararmeters & convergence information
out = struct('params', struct('candidate_lamV', lamV, ...
    'candidate_lamW', lamW, 'candidate_alphaW', alphaW, ...
    'candidate_alphaT', alphaT, 'maxit', maxit, 'warmstart', warmstart, ...
    'threshold', thr, 'K', K), ...
    'optimal_params', struct('lamV', zeros(K,1), 'lamW', zeros(K,1), ...
    'alphaW', zeros(K,1), 'alphaT', zeros(K,1)));
% Initialize factor vectors
Xhat = X;
if warmstart
    % use CP-decomposition as warm start
    U = cp_als(X, K, 'init', 'nvecs');
else
    % use random initialization
    for ii = 1:nD
        U{ii} = rand(size(X,ii), K);
        for kk = 1:K
            U{ii}(:,kk) = U{ii}(:,kk)/norm(U{ii}(:,kk),2);
        end
    end
end

% initialize BIC values
vBIC = 100*ones(length(lamV),K);
wBIC = 100*ones(length(alphaW),length(lamW),K);
tBIC = 100*ones(length(alphaT),1,K);

%% Run nested BIC loop
for kk = 1:K
    if verbose; fprintf('Finding optimal hyperparameters for factor %d out of %d.\n', kk, K); end
    % Reset convergence criteria
    indo = thr + 1; iter = 1;
    while (indo > thr) && (maxit > iter)
        % Store inital factor vectors for k
        oldU = U;
        % (i) Compute factor for trial (1st dim)
        ubrev = double(ttv(ttv(ttv(Xhat, oldU{2}(:,kk), 2), oldU{3}(:,kk), 2), oldU{4}(:,kk), 2));
        norm_ubrev = norm(ubrev,2);
        if norm_ubrev > 0
            uHat = ubrev/norm_ubrev;
        else
            uHat = zeros(n,1);
        end
        U{1}(:,kk) = uHat; clear uHat ubrev
        % (ii) Compute the factor for electrodes (2nd dim)
        vvec = double(ttv(ttv(ttv(Xhat, oldU{1}(:,kk), 1), oldU{3}(:,kk), 2), oldU{4}(:,kk), 2));
        vbrev = repmat(vvec, 1, length(lamV));
        for vv = 1:length(lamV)
            vbrev(:,vv) = soft_thr(vvec, lamV(vv));
            vDF = sum(vbrev(:,vv) ~= 0);
            vBIC(vv,kk) = log((norm(vvec - vbrev(:,vv), 2)^2)/p) + (vDF*log(p)/p);
        end
        % Select model with lowest BIC
        [~, vi] = min(vBIC(:,kk));
        out.optimal_params.lamV(kk) = lamV(vi);
        norm_vbrev = norm(vbrev(:,vi), 2);
        if norm_vbrev > 0
            U{2}(:,kk) = vbrev(:,vi)/norm_vbrev;
        else
            U{2}(:,kk) = zeros(p, 1);
        end
        clear vvec vbrev
        % (iii) Compute the factor for frequency (3rd dim)
        wvec = double(ttv(ttv(ttv(Xhat, oldU{1}(:,kk), 1), oldU{2}(:,kk), 1), oldU{4}(:,kk), 2));
        wbrev = zeros(q, length(alphaW), length(lamW));
        for wa = 1:length(alphaW)
            Sw = speye(q) + alphaW(wa)*omegaW;      % Smoothing matrix for freq
            Lw = eigs(Sw,1);                        % Largest eigenvalue of Sw
            wtild = oldU{3}(:,kk) + (wvec - Sw*oldU{3}(:,kk))/Lw;
            for wl = 1:length(lamW)
                wbrev(:,wa,wl) = soft_thr(wtild, lamW(wl)/Lw);
                actw = wbrev(:,wa,wl) ~= 0;
                wDF = trace(inv(speye(sum(actw)) + alphaW(wl)*omegaW(actw,actw)));
                wBIC(wa,wl,kk) = log((norm(wvec - wbrev(:,wa,wl),2)^2)/q) + (log(q)*wDF/q);
            end
        end
        % Select model with lowest BIC
        M = min(min(wBIC(:,:,kk)));
        [bestA, bestL] = find(wBIC(:,:,kk) == M);        
        out.optimal_params.alphaW(kk) = alphaW(bestA);
        out.optimal_params.lamW(kk) = lamW(bestL);
        Sw = speye(q) + alphaW(bestA)*omegaW;
        norm_wbrev = sqrt(transpose(wbrev(:,bestA,bestL))*Sw*wbrev(:,bestA,bestL));
        if norm_wbrev > 0
            U{3}(:,kk) = wbrev(:,bestA,bestL)/norm_wbrev;
        else
            U{3}(:,kk) = zeros(q,1);
        end
        clear wvec wbrev wtild
        % (iv) Compute the factor for time (4th dim)
        tvec = double(ttv(ttv(ttv(Xhat, oldU{1}(:,kk), 1), oldU{2}(:,kk), 1), oldU{3}(:,kk), 1));
        tbrev = zeros(r, length(alphaT));
        for ta = 1:length(alphaT)
            St = speye(r) + alphaT(ta)*omegaT;
            tbrev(:,ta) = St\tvec;
            tDF = trace(St);
            tBIC(ta,kk) = log((norm(tvec - tbrev(:,ta),2)^2)/r) + (log(r)*tDF/r);
        end
        % Select model with lowest BIC
        [~,ti] = min(tBIC(:,kk));
        out.optimal_params.alphaT(kk) = alphaT(ti);
        St = speye(r) + alphaT(ti)*omegaT;
        tbrev_norm = sqrt(transpose(tbrev(:,ti))*St*tbrev(:,ti));
        if tbrev_norm > 0
            U{4}(:,kk) = tbrev(:,ti)/tbrev_norm;
        else
            U{4}(:,kk) = zeros(r,1);
        end
        clear tvec tbrev
        % update criteria
        iter = iter + 1;
        indo = norm(U{1}(:,kk) - oldU{1}(:,kk),2)/norm(oldU{1}(:,kk),2) + ...
            norm(U{2}(:,kk) - oldU{2}(:,kk),2)/norm(oldU{2}(:,kk),2) + ...
            norm(U{3}(:,kk) - oldU{3}(:,kk),2)/norm(oldU{3}(:,kk),2) + ...
            norm(U{4}(:,kk) - oldU{4}(:,kk),2)/norm(oldU{4}(:,kk),2);
        if verbose; fprintf("Change in convergence is %f.\n", indo); end
    end
    % Repeat, with optimal parameters
    if verbose; fprintf('REPEAT, with optimal params ...\n'); end
    Sw = speye(q) + out.optimal_params.alphaW(kk)*omegaW;     % check for mult q
    Lw = eigs(Sw, 1);
    St = inv(speye(r) + out.optimal_params.alphaT(kk)*omegaT);      % check for mult r
    indo = thr + 1; iter = 1;
    while (indo > thr) && (iter < maxit)
        % Store inital factor vectors for k
        oldU = U;
        % (i) Compute factor for trial (1st dim)
        uvec = double(ttv(ttv(ttv(Xhat, oldU{2}(:,kk), 2), oldU{3}(:,kk), 2), oldU{4}(:,kk),2));
        uvec_norm = norm(uvec, 2);
        if uvec_norm > 0
            U{1}(:,kk) = uvec/uvec_norm;
        else
            U{1}(:,kk) = zeros(n,1);
        end
        clear uvec;
        % (ii) Compute the factor for electrodes (2nd dim)
        vvec = double(ttv(ttv(ttv(Xhat, oldU{1}(:,kk), 1), oldU{3}(:,kk), 2), oldU{4}(:,kk), 2));
        vbrev = soft_thr(vvec, out.optimal_params.lamV(kk));
        vbrev_norm = norm(vbrev, 2);
        if vbrev_norm > 0
            U{2}(:,kk) = vbrev/vbrev_norm;
        else
            U{2}(:,kk) = zeros(p,1);
        end
        clear vbrev vvec;
        % (iii) Compute the factor for frequency (3rd dim)
        wvec = double(ttv(ttv(ttv(Xhat, oldU{1}(:,kk), 1), oldU{2}(:,kk), 1), oldU{4}(:,kk), 2));
        wbrev = oldU{3}(:,kk) + (wvec - Sw*oldU{3}(:,kk))/Lw;
        wbrev = soft_thr(wbrev, out.optimal_params.lamW(kk)/Lw);
        wbrev_norm = sqrt(wbrev'*Sw*wbrev);
        if wbrev_norm > 0
            U{3}(:,kk) = wbrev/wbrev_norm;
        else
            U{3}(:,kk) = zeros(q, 1);
        end
        clear wvec wbrev;
        % Compute the factor for time (4th dim)
        tvec = double(ttv(ttv(ttv(Xhat, oldU{1}(:,kk), 1), oldU{2}(:,kk), 1), oldU{3}(:,kk), 1));
        tbrev = St\tvec;
        tbrev_norm = sqrt(tbrev'*St*tbrev);
        if tbrev_norm > 0
            U{4}(:,kk) = tbrev/tbrev_norm;
        else
            U{4}(:,kk) = zeros(r,1);
        end
        clear tvec tbrev;
        % update criteria
        iter = iter + 1;
        indo = norm(U{1}(:,kk) - oldU{1}(:,kk),2)/norm(oldU{1}(:,kk),2) + ...
            norm(U{2}(:,kk) - oldU{2}(:,kk),2)/norm(oldU{2}(:,kk),2) + ...
            norm(U{3}(:,kk) - oldU{3}(:,kk),2)/norm(oldU{3}(:,kk),2) + ...
            norm(U{4}(:,kk) - oldU{4}(:,kk),2)/norm(oldU{4}(:,kk),2);
        if verbose; fprintf("Change in convergence is %f.\n", indo); end
    end
    % Complete loop and deflate
    dd = ttv(ttv(ttv(ttv(tensor(Xhat),U{1}(:,kk),1),U{2}(:,kk),1),U{3}(:,kk),1),U{4}(:,kk),1);
    Xhat = tensor(Xhat - ktensor(dd, U{1}(:,kk), U{2}(:,kk), U{3}(:,kk), U{4}(:,kk)));
end
% output U
%out.U = U;
%out.BIC = struct('vBIC', vBIC, 'wBIC', wBIC, 'tBIC', tBIC);

%% Turn off parall environment
if par_on; delete(gcp); end

end

