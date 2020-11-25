function [U, fmod, out] = rhopls_fit(X, Y, varargin)
%==========================================================================
% This function performs rho-PLS, and returns i.) reduced form of the
% original data X, and ii.) Fits either an LDA, SVM, or logistic regression
% model (one-class) to the reduced data X and Y.
%
% NOTES
% 1. Assumes that the data X has dimensions (#trials, #electrodes,
% #frequency, #timestamps), which are denoted by (n, p, q, r)
% 2. rhoPLS applies selects influential electrodes with sparsity/L1
% regularization.
% 3. rhoPLS simultaneously selects influential frequencies and smooths over
% them. Frequenices must be sorted in ascending order.
% 4. rhoPLS smooths over time. Time must be sorted in ascending order.
% 5. In this script rhoPLS does one-class classification. However, it could
% be expanded to multi-class classification with any general
% multiclass-classification prediciton model.
%
% INPUT:
%   1. 'X': Tensor with dimensions (n x p x q x r)
%   2. 'Y': Binary matrix with dimensions (n x 1)
%   2. 'K' (default 1): the number of components, K > 0
%   3. 'lamV' (default 0): Positive sparsity penalty for electrode
%   dimension (vector if K > 1)
%   4. 'lamW' (default 0): Positive lasso parameter associated with
%   frequency dimension (vector if K > 1)
%   5. 'alphaW' (default 0): Non-negative multiplier of omegaW (vector if K
%   > 1)
%   6. 'alphaT' (default 0): Non-negative multplier of omegaT (vector if K
%   > 1)
%   7. 'warmstart' (default false): If true, use CP-decomposition to
%   initialize rhoPCA
%   8. 'maxIterations' (default 500): Maximum number of alternating
%   regressions steps
%   9. 'threshold' (default 1e-4): Convergence for change in objective
%   value
%   10. 'verbose' (default true): Boolean value indicating to print all
%   messages in code
%   11. 'model' (default 'lda'): Classificaton model: 'lda' or 'logistic'
%
% OUTPUT:
%   1. 'U': (4 x 1) cell array of (n x K) factors, where n is the 
%       dimensionaltiy of the respective mode
%   2. 'fmod': Fitted model object
%   3. 'out': Metadata, contains parameters and convergence information
%
% Dependencies: Tensor Toolbox package, rhopca4rhopls.m
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
default_model = 'lda';
% Parse input parameters
params = inputParser;
params.CaseSensitive = true;
f = @(ii) isa(ii,'double')||isa(ii,'tensor');
addRequired(params, 'X', f);
addRequired(params, 'Y', f);
addParameter(params, 'K', default_K, @(ii) (isnumeric(ii)&&(ii > 0)))
addParameter(params, 'lamV', default_lam, @(ii) all(isnumeric(ii)&(ii>=0)));
addParameter(params, 'lamW', default_lam, @(ii) all(isnumeric(ii)&(ii>=0)));
addParameter(params, 'alphaW', default_alpha, @(ii) all(isnumeric(ii)&(ii>=0)));
addParameter(params, 'alphaT', default_alpha, @(ii) all(isnumeric(ii)&(ii>=0)));
addParameter(params, 'maxIterations', default_maxit, @(ii) isnumeric(ii)&&(ii>1));
addParameter(params, 'threshold', default_threshold, @(ii) isnumeric(ii)&&(ii>0));
addParameter(params, 'verbose', default_verbose, @islogical);
addParameter(params, 'warmstart', default_warmstart, @islogical);
addParameter(params, 'model', default_model, @ischar);
parse(params, X, Y, varargin{:});
K = params.Results.K;
lamV = params.Results.lamV;
lamW = params.Results.lamW;
alphaW = params.Results.alphaW;
alphaT = params.Results.alphaT;
maxit = params.Results.maxIterations;
verbose = params.Results.verbose;
warmstart = params.Results.warmstart;
thr = params.Results.threshold;
model = params.Results.model;
% Check dimensionality of covariates X
X = tensor(squeeze(X));         % Remove singletion dimensions, if present
nDim = length(size(X));
if nDim ~= 4; error('Data must be a 4-dimensional tensor'); end
% Check dimensionality of response Y
Y = double(squeeze(Y));
if size(X, 1) ~= size(Y, 1); Y = transpose(Y); end
if size(X,1) ~= size(Y,1)
    error('The arrays X and Y must match on the 1st dimension.'); 
end
if size(Y,2) > 1; error('Y must be single class'); end
% Check model type
model = lower(model);
if ~any(strcmp(model, {'lda','logreg'}))
    error("The only models that may be specified are 'lda' or 'logreg'.\n");
end
% Check length of K and parameters
if K ~= length(lamV)
    warning('K=%d does not match length of either lamV (len=%d). Parameter values will either be replicated or discarded to match K.', K, length(lamV));
    if length(lamV) < K; lamV = repelem(lamV, K); end
    lamV = lamV(1:K);
end  
if K ~= length(lamW)
    warning('K=%d does not match length of either lamW (len=%d). Parameter values will either be replicated or discarded to match K.', K, length(lamW));
    if length(lamW) < K; lamW = repelem(lamW, K); end
    lamW = lamW(1:K);
end  
if K ~= length(alphaW)
    warning('K=%d does not match length of either alphaW (len=%d). Parameter values will either be replicated or discarded to match K.', K, length(alphaW));
    if length(alphaW) < K; alphaW = repelem(alphaW, K); end
    alphaW = alphaW(1:K);
end  
if K ~= length(alphaT)
    warning('K=%d does not match length of either alphaT (len=%d). Parameter values will either be replicated or discarded to match K.', K, length(alphaT));
    if length(alphaT) < K; alphaT = repelem(alphaT, K); end
    alphaT = alphaT(1:K);
end  

%% Run rho-PLS
% Center Y
Ybar = Y - mean(Y);
% Compute tensor covariance of X and Y
if verbose; fprintf('Computing covariance tensor\n'); end
% Run rho-PCA
[U, out] = rhopca4rhopls(squeeze(ttm(X, Ybar, 1, 't')), 'K', K, ...
    'maxIterations', maxit, 'threshold', thr, 'verbose', verbose, ...
    'warmstart', warmstart, ...
    'lamV', lamV, 'lamW', lamW, 'alphaW', alphaW, 'alphaT', alphaT);

%% Fit model
tic;
% Reduce the data
Xtilde = nan(size(X,1), K);
for kk = 1:K
    Xtilde(:,kk) = double(ttv(ttv(X,U{3}(:,kk),4),U{2}(:,kk),3))*U{1}(:,kk);
end
% fit model
out.model = model;      % save model type
if strcmp(model, 'lda')
    fmod = fitcdiscr(Xtilde, Y, 'DiscrimType', 'pseudolinear');
elseif strcmp(model, 'logreg')
    fmod = fitglm(Xtilde, Y, 'linear', 'Distribution', 'binomial');
else
    error("Cannot recognize model; try 'lda' or 'logreg'\n");
end
fit_runtime = toc;
out.convergence.fit_runtime = fit_runtime + out.convergence.runtime;
end

function [U, out] = rhopca4rhopls(Z, varargin)
%==========================================================================
% Computes the K sparse higher order orthogonal iteration factors from
% a tensor covariance matrix (X x_1 Y). 
%
% NOTES
% 1. rhoPCA applies selects influential electrodes with sparsity/L1
%       regualrization.
% 2. rhoPCA simultaneously selects influential frequencies and smooths over
%       them. Frequenices must be sorted in ascending order.
% 3. rhoPCA smooths over time. Time must be sorted in ascending order.
%
% INPUT:
%   1. 'Z': An 3-dimensional tensor 
%   3. 'K' (default 1): the number of components, K > 0
%   4. 'lamV' (default 0): Positive sparsity penalty for 
%       electrode dimension (vector if K > 1)
%   5. 'lamW' (default 0): Positive lasso parameter associated 
%       with frequency dimension (vector if K > 1)
%   6. 'alphaW' (default 0): Non-negative multiplier of omegaW (vector if 
%       K > 1)
%   7. 'alphaT' (default 0): Non-negative multplier of omegaT (vector if 
%       K > 1)
%   8. 'warmstart' (default true): Optional, if true, use 
%       CP-decomposition to initialize rhoPCA
%   9. 'maxIterations' (default 500): Optional, maximum number of 
%       alternating regressions steps
%   10. 'threshold' (default 1e-4): Optional, convergence for change in 
%       objective value
%   11. 'verbose' (default true): Optional, boolean value indicating to 
%       print all messages in code
%
% OUTPUT:
%   1. 'U': (4 x 1) cell array of (n x K) factors, where n is the
%      dimensionaltiy of the respective mode
%   2. 'D': (K x 1) vector of scale parameters
%   3. 'Xhat': A 4-D tensor estimate of X
%   3. 'out': Metadata, contains parameters and convergence information
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
% Parse input parameters
params = inputParser;
params.CaseSensitive = true;
addRequired(params, 'Z', @(ii) isa(ii,'double')||isa(ii,'tensor'));
addParameter(params, 'K', default_K, @(ii) (isnumeric(ii)&&(ii > 0)));
addParameter(params, 'lamV', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'lamW', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'alphaW', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'alphaT', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'maxIterations', default_maxit, @(ii) isnumeric(ii)&&(ii>1));
addParameter(params, 'threshold', default_threshold, @(ii) isnumeric(ii)&&(ii>0));
addParameter(params, 'verbose', default_verbose, @islogical);
addParameter(params, 'warmstart', default_warmstart, @islogical);
parse(params, Z, varargin{:});
K = params.Results.K;
lamV = params.Results.lamV;
lamW = params.Results.lamW;
alphaW = params.Results.alphaW;
alphaT = params.Results.alphaT;
maxit = params.Results.maxIterations;
verbose = params.Results.verbose;
warmstart = params.Results.warmstart;
thr = params.Results.threshold;
% Check dimensionality of covariates X
Z = squeeze(Z);         % Remove singletion dimensions, if present
nD = length(size(Z));
if nD ~= 3; error('Data must be a 3-dimensional tensor\n.'); end
% Convert X to tensor class
if ~isa(Z, 'tensor'); Z = tensor(Z); end

%% Set up values for rho-PCA
% Load dimensions
p = size(Z,1); q = size(Z,2); r = size(Z,3);
%nDims = size(Z); p = nDims(1); q = nDims(2); r = nDims(3);
% Set both omegaW and omegaT to be 2nd differencing matrices
omegaT = gallery('tridiag', r);
omegaW = gallery('tridiag', q);
% Store pararmeters & convergence information
out = struct('params', struct('lamV', lamV, 'K', K, 'lamW', lamW, ...
    'alphaW', alphaW, 'alphaT', alphaT, 'maxit', maxit, ...
    'warmstart', warmstart, 'threshold', thr), ...
    'convergence', struct('objectiveVals', [], 'runtime', 0));
% Initialize factor vectors
if warmstart
    % use CP-decomposition as warm start
    U = cp_als(Z, K, 'init', 'nvecs');
    D = U.lambda;
else
    % use random initialization
    U = cell(nD,1);
    for ii = 1:nD
        U{ii} = rand(size(Z,ii), K);
        for kk = 1:K; U{ii}(:,kk) = U{ii}(:,kk)/norm(U{ii}(:,kk),2); end
    end
    D = rand(K,1);
end
% Compute objective value
out.convergence.objectiveVals(end+1) = -1000;

%% Run rho-PLS
tic;
for kk = 1:K
    if verbose; fprintf('Computing factor %d\n', kk); end
    % Pre-compute parameters
    Sw = eye(q) + alphaW(kk)*omegaW;    % Smoothing matrix for frequency 
    St = eye(r) + alphaT(kk)*omegaT;    % Smoothing matrix for time
    Lw = eigs(Sw, 1);                   % Largest eigenvalue of Sw
    % Reset convergence critera
    iter = 1; deltaObj = thr + 1;     
    while (deltaObj > thr) && (maxit > iter)
        % Store initial factor vectors for k
        oldU = U;
        % (i) Compute CP vectors for electrode (1st dim)
        vvec = double(ttv(Z, oldU{3}(:,kk), 3))*oldU{2}(:,kk); 
        vvec = soft_thr(vvec, lamV(kk));
        vvec_norm = norm(vvec, 2);
        if vvec_norm > 0
            U{1}(:,kk) = vvec/vvec_norm;
        else
            U{1}(:,kk) = zeros(p,1);
        end
        % (ii) Compute CP vector for frequency (2nd dimension)
        wvec = double(ttv(Z, oldU{3}(:,kk), 3))'*oldU{1}(:,kk); 
        wvec = oldU{2}(:,kk) + (wvec - (Sw*oldU{2}(:,kk)))/Lw;
        wvec = soft_thr(wvec, lamW(kk)/Lw); 
        wvec_norm = sqrt(transpose(wvec)*Sw*wvec);
        if wvec_norm > 0
            U{2}(:,kk) = wvec/wvec_norm;
        else
            U{2}(:,kk) = zeros(q,1);
        end
        % (iii) Compute CP vector for time (3rd dimension)
        tvec = double(ttv(ttv(Z, oldU{1}(:,kk), 1), oldU{2}(:,kk), 1));
        tvec = St\tvec;
        tvec_norm = sqrt(transpose(tvec)*St*tvec);
        if tvec_norm > 0
            U{3}(:,kk) = tvec/tvec_norm;
        else
            U{3}(:,kk) = zeros(r,1);
        end
       % Compute objective value
       obj = oldU{1}(:,kk)'*double(ttv(Z, oldU{3}(:,kk), 3))*oldU{2}(:,kk) - ...
           lamV(kk)*norm(U{1}(:,kk), 1) - lamW(kk)*norm(U{2}(:,kk),1);
       out.convergence.objectiveVals(end+1) = obj;
       % Record changes in objective function
       deltaObj = abs(out.convergence.objectiveVals(end) - out.convergence.objectiveVals(end-1));
       if verbose; fprintf('\tAbsolute change in objective value: %f\n', deltaObj); end
       % Prepare for next iteration of finding CP components of k
       iter = iter + 1;
    end
    % Compute CP-constant for factor k
    D(kk) = oldU{1}(:,kk)'*double(ttv(Z, oldU{3}(:,kk), 3))*oldU{2}(:,kk);
    Z = deflate(Z, U, D, kk);
end
out.convergence.runtime = toc;
end

function X = deflate(X, U, D, kk)
%==========================================================================
% This function performs the deflation step
% Avoid copy-on-write behavior
%==========================================================================
    X = tensor(X - ktensor(D(kk), U{1}(:,kk), U{2}(:,kk), U{3}(:,kk)));
end
