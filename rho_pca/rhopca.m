function [U, D, X, out] = rhopca(X, varargin)
%==========================================================================
% Computes the K sparse higher order orthogonal iteration factors from
% tensor X with dimsensions (#trial x #electrode x #frequency x timestamps)
%
% NOTES
% 1. rhoPCA applies selects influential electrodes with sparsity/L1
%       regualrization.
% 2. rhoPCA simultaneously selects influential frequencies and smooths over
%       them. Frequenices must be sorted in ascending order.
% 3. rhoPCA smooths over time. Time must be sorted in ascending order.
%
% INPUT:
%   1. 'X': An 4-dimensional tensor
%   2. 'K' (default 1): the number of components, K > 0
%   3. 'lamV' (default 0): Positive sparsity penalty for electrode
%       dimension (vector if K > 1)
%   4. 'lamW' (default 0): Positive lasso parameter associated with
%       frequency dimension (vector if K > 1)
%   5. 'alphaW' (default 0): Non-negative multiplier of omegaW (vector if
%       K > 1)
%   6. 'alphaT' (default 0): Non-negative multplier of omegaT (vector if
%       K > 1)
%   7. 'warmstart' (default true): Optional, if true, use CP-decomposition
%       to initialize rhoPCA
%   8. 'maxIterations' (default 500): Optional, maximum number of
%       alternating regressions steps
%   9. 'threshold' (default 1e-4): Optional, convergence for change in
%       objective value
%   10. 'verbose' (default true): Optional, boolean value indicating to
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
addRequired(params, 'X', @(ii) isa(ii,'double')||isa(ii,'tensor'));
addParameter(params, 'K', default_K, @(ii) (isnumeric(ii)&&(ii > 0)))
addParameter(params, 'lamV', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'lamW', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'alphaW', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'alphaT', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'maxIterations', default_maxit, @(ii) isnumeric(ii)&&(ii>1));
addParameter(params, 'threshold', default_threshold, @(ii) isnumeric(ii)&&(ii>0));
addParameter(params, 'verbose', default_verbose, @islogical);
addParameter(params, 'warmstart', default_warmstart, @islogical);
parse(params, X, varargin{:});
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
X = double(squeeze(X));         % Remove singletion dimensions, if present
nD = length(size(X));
if nD ~= 4; error('Data must be a 4-dimensional tensor'); end
% Convert X to tensor class
if ~isa(X, 'tensor'); X = tensor(X); end
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

%% Set up rho-PCA
% Load dimensions
nD = length(size(X)); 
n = size(X,1); p = size(X,2); q = size(X,3); r = size(X,4); 
% Set both omegaW and omegaT to be 2nd differencing matrices
omegaT = gallery('tridiag', r);
omegaW = gallery('tridiag', q);
% Store pararmeters & convergence information
out = struct('params', struct('lamV', lamV, 'lamW', lamW, 'K', K, ...
    'alphaW', alphaW, 'alphaT', alphaT, 'maxit', maxit, ...
    'warmstart', warmstart, 'threshold', thr), ...
    'convergence', struct('objectiveVals', [], 'runtime', 0));
% Initialize factor vectors
if warmstart
    % use CP-decomposition as warm start
    U = cp_als(X, K, 'init', 'nvecs');
    D = U.lambda;
else
    % use random initialization
    U = cell(nD,1);
    for ii = 1:nD
        U{ii} = rand(size(X,ii), K);
        for kk = 1:K; U{ii}(:,kk) = U{ii}(:,kk)/norm(U{ii}(:,kk),2); end
    end
    D = rand(K,1);
end
% Compute objective value
out.convergence.objectiveVals(end+1) = -1000;

%% Run rho-PCA
tic;                    % time rhoPCA
for kk = 1:K
    if verbose; fprintf('Computing factor %d\n', kk); end
    % Pre-compute parameters
    Sw = eye(q) + alphaW(kk)*omegaW;    % Smoothing matrix for frequency
    St = eye(r) + alphaT(kk)*omegaT;    % Smoothing matrix for time
    Lw = eigs(Sw, 1);                   % Largest eigenvalue of Sw
    % Reset convergence critera
    iter = 1; deltaObj = thr + 1;
    % Find u_k, v_k, w_k, t_k
    while (deltaObj > thr) && (maxit > iter)
        % Store initial factor vectors for k
        oldU = U;
        % (i) Compute CP vector for trial (1st dim)
        uvec = double(ttv(ttv(X, oldU{4}(:,kk), 4), oldU{3}(:,kk), 3))*oldU{2}(:,kk); 
        uvec_norm = norm(uvec, 2);
        if uvec_norm > 0
            U{1}(:,kk) = uvec/uvec_norm;
        else
            U{1}(:,kk) = zeros(n,1);
        end
        clear uvec
        % (ii) Compute CP vectors for electrode (2nd dim)
        vvec = transpose(oldU{1}(:,kk)'*double(ttv(ttv(X, oldU{4}(:,kk), 4), oldU{3}(:,kk), 3)));
        vvec = soft_thr(vvec, lamV(kk));
        vvec_norm = norm(vvec, 2);
        if vvec_norm > 0
            U{2}(:,kk) = vvec/vvec_norm;
        else
            U{2}(:,kk) = zeros(p,1);
        end
        clear vvec
        % (iii) Compute CP vector for frequency (3rd dimension)
        wvec = transpose(oldU{2}(:,kk)'*double(ttv(ttv(X, oldU{4}(:,kk), 4), oldU{1}(:,kk), 1)));
        wvec = oldU{3}(:,kk) + (wvec - (Sw*oldU{3}(:,kk)))/Lw;
        wvec = soft_thr(wvec, lamW(kk)/Lw);
        wvec_norm = sqrt(transpose(wvec)*Sw*wvec); 
        if wvec_norm > 0
            U{3}(:,kk) = wvec/wvec_norm;
        else
            U{3}(:,kk) = zeros(q,1);
        end
        clear wvec
        % (iv) Compute CP vector for time (4th dimension)
        tvec = double((ttv(ttv(X, oldU{1}(:,kk), 1), oldU{2}(:,kk), 1)))'*oldU{3}(:,kk);
        tvec = St\tvec;
        tvec_norm = sqrt(transpose(tvec)*St*tvec);
        if tvec_norm > 0
            U{4}(:,kk) = tvec/tvec_norm;
        else
            U{4}(:,kk) = zeros(r,1);
        end
        clear tvec
        % Compute objective value
        obj = oldU{1}(:,kk)'*double(ttv(ttv(X, oldU{4}(:,kk), 4), oldU{3}(:,kk), 3))*oldU{2}(:,kk) - ...
            lamV(kk)*norm(U{2}(:,kk),1) - lamW(kk)*norm(U{3}(:,kk),1);
        out.convergence.objectiveVals(end+1) = obj;
        % Record changes in objective function
        deltaObj = abs(out.convergence.objectiveVals(end) - out.convergence.objectiveVals(end-1));
        if verbose; fprintf('\tAbsolute change in objective value: %f\n', deltaObj); end
        % Prepare for next iteration of finding CP components of k
        iter = iter + 1;
    end
    % Compute CP-constant for factor k
    D(kk) = oldU{1}(:,kk)'*double(ttv(ttv(X, oldU{4}(:,kk), 4), oldU{3}(:,kk), 3))*oldU{2}(:,kk);
    X = deflate(X, U, D, kk);
end
out.convergence.runtime = toc;
end

function X = deflate(X, U, D, kk)
%==========================================================================
% This function performs the deflation step
% Avoid copy-on-write behavior
%==========================================================================
    X = tensor(X - ktensor(D(kk), U{1}(:,kk), U{2}(:,kk), U{3}(:,kk), U{4}(:,kk)));
end
