function [out] = tune_rhopls(X, Y, parts, varargin)
% This function tunes rhopPLS + classifier by selecting the optimal
% lambda_v, lambda_w, alpha_w, and alpha_t for each k = 1, ..., K
% components. Parameters are selected by a nested B-fold cross-validation.
%
% INPUT:
%   1. 'X': An 4-dimensional tensor with dimensions (n x p x q x r)
%   2. 'Y': An matrix with dimensions (n x g)
%   3. 'parts': Cell array containing paritions of data, from 'cvpartition'
%   4. 'K' (default 1): the number of components, K > 0
%   5. 'candidate_lamV' (default 0.1): Vector of positive sparsity penalty
%   for electrode dimension (vector if K > 1)
%   6. 'candidate_lamW' (default 0,1): Vector if positive lasso parameter
%   associated with frequency dimension (vector if K > 1)
%   7. 'candidate_alphaW' (default 0.1): Vector of non-negative multiplier
%   ofomegaW (vector if K > 1)
%   8. 'candidate_alphaT' (default 0.1): Non-negative multplier of omegaT
%   (vector if K > 1)
%   9. 'warmstart' (default true): Optional, if true, use CP-decomposition
%   to initialize rhoPCA
%   10. 'maxIterations' (default 1000): Optional, maximum number of 
%       alternating regressions steps
%   11. 'threshold' (default 1e-6): Optional, convergence for change in 
%       objective value
%   12. 'verbose' (default true): Optional, boolean value indicating to 
%       print all messages in code
%   13. 'model' (default 'lda'): Classificaton model: 'lda', 'svm', or
%   'logistic'
%   14. 'Nu' (default 0.5): Only for SVM - Nu controls the tradeoff between
%   ensuring that most training examples are in the positive class and
%   minimizing the weights in the score function; must be greater than 0
%   and at most 1
%
% OUTPUT:
%   1. 'out': Metadata, contains parameters and convergence information
%
% Dependencies: Tensor Toolbox package, rhopls_fit.m, rhopls_predict.m
%==========================================================================

%% Load and verify parameters
% Specify default parameters and their expected values
default_K = 1;
default_maxit = 500;
default_threshold = 1e-4;
default_lam = 0.1;
default_alpha = 0.1;
default_verbose = true;
default_warmstart = true;
default_nu = 0.5;
default_model = 'lda';
% Parse input parameters
params = inputParser;
params.CaseSensitive = true;
addRequired(params, 'X', @(ii) isa(ii,'double')||isa(ii,'tensor'));
addRequired(params, 'Y', @(ii) isa(ii,'double'));
addRequired(params, 'parts', @(ii) isa(ii,'cell'));
addParameter(params, 'K', default_K, @(ii) (isnumeric(ii)&&(ii > 0)))
addParameter(params, 'candidate_lamV', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'candidate_lamW', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'candidate_alphaW', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'candidate_alphaT', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'maxIterations', default_maxit, @(ii) isnumeric(ii)&&(ii>1));
addParameter(params, 'threshold', default_threshold, @(ii) isnumeric(ii)&&(ii>0));
addParameter(params, 'verbose', default_verbose, @islogical);
addParameter(params, 'warmstart', default_warmstart, @islogical);
addParameter(params, 'model', default_model, @ischar);
addParameter(params, 'Nu', default_nu, @(ii)(ii>0)&(ii<=1));
parse(params, X, Y, parts, varargin{:});
K = params.Results.K;
lamV = params.Results.candidate_lamV;
lamW = params.Results.candidate_lamW;
alphaW = params.Results.candidate_alphaW;
alphaT = params.Results.candidate_alphaT;
maxit = params.Results.maxIterations;
verbose = params.Results.verbose;
warmstart = params.Results.warmstart;
thr = params.Results.threshold;
nu = params.Results.Nu;
model = params.Results.model;
% Check dimensionality of covariates X
X = double(squeeze(X));         % Remove singletion dimensions, if present
if length(size(X)) ~= 4; error('Data X must be a 4-dimensional tensor'); end
% Check dimensionality of response Y
Y = double(squeeze(Y));
if size(X,1) ~= size(Y,1)
    error('The arrays X and y must match on the 1st dimension.'); 
end
if size(Y,2) > 1; error('y must be a vector for one-class classification\n.'); end
% Check model type
model = lower(model);
if ~any(strcmp(model, {'lda','svm','logreg'}))
    error("The only models that may be specified are 'lda', 'svm' or 'logreg'.\n");
end

%% Set up variables
% Load dimensions
nD = length(size(X));
n = size(X, 1);
% Store pararmeters & convergence information
out = struct('params', struct('maxit', maxit, 'warmstart', warmstart, ...
    'threshold', thr, 'K', K, 'model', model, 'Nu', nu), ...
    'optimal_params', struct('lamV', zeros(K,1), 'lamW', zeros(K,1), ...
    'alphaW', zeros(K,1), 'alphaT', zeros(K,1)), ...
    'candidate_params', struct('candidate_lamV', lamV, ...
    'candidate_lamW', lamW, 'candidate_alphaW', alphaW, ...
    'candidate_alphaT', alphaT));    

%% Initialization
% We hold out the first partition as a test set. It's ignored in this
% function, and the other parts are used as training and validation sets.
% We plug in all of X so that the dimension align -- the first partition is
% still held out though. See 'full_prediction_pipeline.m' for details.
nFolds = length(parts);
allSets = 1:nFolds;
% Store data for each fold
foldData = cell(nFolds, 1);
for bb = 1:nFolds; foldData{bb} = struct(); end
% Make a grid of parameters
J = length(lamV)*length(lamW)*length(alphaW)*length(alphaT);
candidate_params = zeros(J, 4);
hh = 1;
for ii = 1:length(lamV)
    for jj = 1:length(lamW)
        for kk = 1:length(alphaW)
            for ll = 1:length(alphaT)
                candidate_params(hh, 1) = lamV(ii);
                candidate_params(hh, 2) = lamW(jj);
                candidate_params(hh, 3) = alphaW(kk);
                candidate_params(hh, 4) = alphaT(ll);
                hh = hh + 1;
            end
        end
    end
end

%% Begin nested B-fold cross-validation
% Step 2
for kk = 1:K
    if verbose; fprintf('Computing component %d of %d.\n', kk, K); end    
    % Store accuracy
    phi_k = zeros(J,nFolds);
    % Step 2(A)
    for bb = 1:nFolds
        % Define Xtest, Xtrain, Ytrain, Ytest
        VAL = logical(parts{bb});
        TR = zeros(n, 1);
        trSets = allSets(~ismember(allSets, bb));
        for ii = 1:(nFolds-1); TR = logical(TR + parts{trSets(ii)}); end
        Xtest = tensor(X(VAL,:,:,:));
        Ytest = Y(VAL);
        Xtrain = tensor(X(TR,:,:,:));
        Ytrain = Y(TR);
        % Initialization for kk == 1
        if kk == 1
            % Calculate the covariance tensor
            Ybar = Ytrain - mean(Ytrain);
            if verbose; fprintf('Computing covariance tensor\n'); end
            Zb = squeeze(ttm(Xtrain, Ybar, 1, 't'));
            foldData{bb}.Zhat = Zb;
            % Initialize factor vectors
            if warmstart
                % use CP-decomposition as warm start
                U = cp_als(tensor(Zb), K, 'init', 'nvecs');
                D = U.lambda;
            else
                % use random initialization
                U = cell(3, 1);
                for ii = 1:3
                    U{ii} = rand(size(Zb,ii), K);
                    for qq = 1:K; U{ii}(:,qq) = U{ii}(:,qq)/norm(U{ii}(:,qq),2); end
                end
                D = rand(K,1);
            end
            foldData{bb}.U = U;
            foldData{bb}.D = D;
            % Store previous estimates of u,v,w,t
            foldData{bb}.oldU = U;
        end
        % Step 2A(ii): For each set of candidate parameters
        for jj = 1:J
            if verbose; fprintf('\tTry candidate parameters %d of %d.\n', jj, J); end 
            % Step 2A(ii)(b): perform k-step of rho-PLS
            [Ujbk, ~, ~] = kstep_rhopls(foldData{bb}.Zhat, ...
                foldData{bb}.U, kk, foldData{bb}.D, ...
                'maxIterations', maxit, 'verbose', false, ...
                'threshold', thr, ...
                'lamV', out.optimal_params.lamV(kk), ...
                'lamW', out.optimal_params.lamW(kk), ...
                'alphaW', out.optimal_params.alphaW(kk), ...
                'alphaT', out.optimal_params.alphaT(kk));
            % Step 2A(ii)(c): reduce the dimensionality of the data
            Xtilde_tr = double(ttv(ttv(ttv(tensor(Xtrain), Ujbk{1}(:,kk), 2), Ujbk{2}(:,kk), 2), Ujbk{3}(:,kk), 2));
            Xtilde_ts = double(ttv(ttv(ttv(tensor(Xtest), Ujbk{1}(:,kk), 2), Ujbk{2}(:,kk), 2), Ujbk{3}(:,kk), 2));
            % Step 2A(ii)(d): Predict the class membership 
            if strcmp(model, 'lda')
                fmod = fitcdiscr(Xtilde_tr, Ytrain, 'DiscrimType', 'pseudolinear');
            elseif strcmp(model, 'svm')
                fmod = fitcsvm(Xtilde_tr, Ytrain, 'ClassNames', [0,1], 'Nu', nu);
            elseif strcmp(model, 'logreg')
                fmod = fitglm(Xtilde_tr, Ytrain, 'linear', 'Distribution', 'binomial');
            else
                error("Cannot recognize model; try 'lda', 'svm', or 'logreg'\n");
            end
            % Predict the class membership of Ytest
            Yhat = predict(fmod, Xtilde_ts);
            if length(unique(Yhat)) > 2; Yhat = Yhat >= 0.5; end
            % Step 2A(ii)(e) Calculate the accuracy
            phi_k(jj,bb) = calc_accuracy(Ytest, Yhat);
        end
    end  
    % Step 2(B): Find the optimal parameter set for k, for fold bb
    [~, jstar] = max(mean(phi_k, 2));
    % store optimal parameter set for k
    out.optimal_params.lamV(kk) = candidate_params(jstar,1);
    out.optimal_params.lamW(kk) = candidate_params(jstar,2);
    out.optimal_params.alphaW(kk) = candidate_params(jstar,3);
    out.optimal_params.alphaT(kk) = candidate_params(jstar,4);
    % Step 2(C): For each fold b = 1... nFolds
    for bb = 1:nFolds
        % Step 2C(i): Cache the component vector estimates for jstar by 
        % performing the k-step of rho-PLS, 
        % Step 2C(ii) deflate the Zbk tensor (included in kstep_rhopls)
        [Ubk, Dbk, Zbk] = kstep_rhopls(foldData{bb}.Zhat, foldData{bb}.U, ...
                kk, foldData{bb}.D, ...
                'maxIterations', maxit, 'verbose', false, ...
                'threshold', thr, ...
                'lamV', out.optimal_params.lamV(kk), ...
                'lamW', out.optimal_params.lamW(kk), ...
                'alphaW', out.optimal_params.alphaW(kk), ...
                'alphaT', out.optimal_params.alphaT(kk));
        % update fold information
        foldData{bb}.oldU = foldData{bb}.U;
        foldData{bb}.U = Ubk;
        foldData{bb}.D = Dbk;
        foldData{bb}.Zhat = Zbk;
    end
end
end

function [U, D, Zhat] = kstep_rhopls(Zhat, U, kk, D, varargin)
% This function calls one single k-step for tuning rho-PCA, used in the
% nested B-fold cross-validation schema. 
%
% This also contains the deflation step.
%
% INPUT:
%   1. 'U': initial factor estimates, cell array of vectors
%   2. 'D': initial factor constant
%   3. 'Zhat': An 3-dimensional tensor with dimensions (p x q x r)
%   4. 'lamV' (default 0.1): positive sparsity penalty for electrode
%   dimension
%   5. 'lamW' (default 0.1): positive lasso parameter associated with
%   requency dimension
%   6. 'alphaW' (default 0.1): non-negative multiplier of omegaW 
%   7. 'alphaT' (default 0.1): Non-negative multplier of omegaT
%   8. 'maxIterations' (default 1000): Optional, maximum number of 
%       alternating regressions steps
%   9. 'threshold' (default 1e-6): Optional, convergence for change in 
%       objective value
%   10. 'verbose' (default true): Optional, boolean value indicating to 
%       print all messages in code
%
% OUTPUT:
%   1. 'optParams': structure containing the optimal parameters selected.
%   1. 'U': (4 x 1) cell array of (n x K) factors, where n is the
%      dimensionaltiy of the respective mode
%   2. 'D': (K x 1) vector of scale parameters
%   3. 'Zhat': A 4-D tensor estimate of X
%   3. 'out': Metadata, contains parameters and convergence information
%
% Dependencies: Tensor Toolbox package
%==========================================================================

%% Load and verify parameters
% Specify default parameters and their expected values
default_maxit = 500;
default_threshold = 1e-4;
default_lam = 0;
default_alpha = 0;
default_verbose = true;
% Parse input parameters
params = inputParser;
params.CaseSensitive = true;
addRequired(params, 'Zhat', @(ii) isa(ii,'double')||isa(ii,'tensor'));
%addRequired(params, 'U', @(ii) isa(ii,'cell'));
addRequired(params, 'U');
addRequired(params, 'kk', @isnumeric);
addRequired(params, 'D', @isnumeric);
addParameter(params, 'lamV', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'lamW', default_lam, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'alphaW', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'alphaT', default_alpha, @(ii) all((isnumeric(ii)&(ii>=0))));
addParameter(params, 'maxIterations', default_maxit, @(ii) isnumeric(ii)&&(ii>1));
addParameter(params, 'threshold', default_threshold, @(ii) isnumeric(ii)&&(ii>0));
addParameter(params, 'verbose', default_verbose, @islogical);
parse(params, Zhat, U, kk, D, varargin{:});
lamV = params.Results.lamV;
lamW = params.Results.lamW;
alphaW = params.Results.alphaW;
alphaT = params.Results.alphaT;
maxit = params.Results.maxIterations;
verbose = params.Results.verbose;
thr = params.Results.threshold;
% Check dimensionality of covariance tensor Z
Zhat = double(squeeze(Zhat));      
nD = length(size(Zhat));
if nD ~= 3; error('Covariance tensor must be a 3-dimensional tensor'); end
if ~isa(Zhat, 'tensor'); Zhat = tensor(Zhat); end

%% Set up variables
% Load dimensions
nD = size(Zhat); p = nD(1); q = nD(2); r = nD(3);
% Set both omegaW and omegaT to be 2nd differencing matrices
omegaT = gallery('tridiag', r);
St = eye(r) + alphaT*omegaT;
omegaW = gallery('tridiag', q);
Sw = eye(q) + alphaW*omegaW;
Lw = eigs(Sw, 1);
% Store pararmeters & convergence information
out = struct('convergence', struct('objectiveVals', [], 'runtime', 0));
% Compute objective value
obj = rhopls_compute_objective_value(Zhat, U, kk, lamV, lamW);
out.convergence.objectiveVals(end+1) = obj;

%% Perform k-step of rho-PLS to find v_k, w_k, and t_k
iter = 1; deltaObj = thr + 1;     % Absolute change in objective value
while (deltaObj > thr) && (maxit > iter)
    oldU = U;
    % (i) Compute CP vectors for electrode (1st dim)
    vvec = double(ttv(ttv(Zhat, oldU{2}(:,kk), 2), oldU{3}(:,kk), 2));
    vvec = soft_thr(vvec, lamV);
    vvec_norm = norm(vvec, 2);
    if vvec_norm > 0
        U{1}(:,kk) = vvec/vvec_norm;
    else
        U{1}(:,kk) = zeros(p,1);
    end
    % (ii) Compute CP vector for frequency (2nd dimension)
    wvec = double(ttv(ttv(Zhat, oldU{1}(:,kk), 1), oldU{3}(:,kk), 2));
    wvec = oldU{2}(:,kk) + (wvec - (Sw*oldU{2}(:,kk)))/Lw;
    wvec = soft_thr(wvec, lamW/Lw);
    wvec_norm = sqrt(transpose(wvec)*Sw*wvec);
    if wvec_norm > 0
        U{2}(:,kk) = wvec/wvec_norm;
    else
        U{2}(:,kk) = zeros(q,1);
    end
    % (iii) Compute CP vector for time (3rd dimension)
    tvec = double(ttv(ttv(Zhat, oldU{1}(:,kk), 1), oldU{2}(:,kk), 1));
    tvec = St\tvec;
    tvec_norm = sqrt(transpose(tvec)*St*tvec);
    if tvec_norm > 0
        U{3}(:,kk) = tvec/tvec_norm;
    else
        U{3}(:,kk) = zeros(r,1);
    end
    % Compute objective value
    obj = rhopls_compute_objective_value(Zhat, U, kk, lamV, lamW);
    out.convergence.objectiveVals(end+1) = obj;
    % Record changes in objective function
    deltaObj = abs(out.convergence.objectiveVals(end) - out.convergence.objectiveVals(end-1));
    if verbose; fprintf('\tAbsolute change in objective value: %f\n', deltaObj); end
    % Prepare for next iteration of finding CP components of k
    iter = iter + 1;
end
% Deflation step
D = ttv(ttv(ttv(tensor(Zhat), U{1}(:,kk), 1), U{2}(:,kk), 1), U{3}(:,kk), 1);
Zhat = tensor(Zhat - ktensor(D, U{1}(:,kk), U{2}(:,kk), U{3}(:,kk)));
end

function [obj] = rhopls_compute_objective_value(Zhat, U, kk, lamV, lamW)
%==========================================================================
% This function computes the objective value
%
% INPUT:
%   1. Zhat - N-dimensional tensor
%   2. u - Cell array of vectors from CP decompostion
%   3. lamV - L1 penalty for electrodes
%   4. lamW - L1 penalty for frequency
%
% OUTPUT:
%   1. obj - objective value
%==========================================================================
obj = ttv(ttv(ttv(tensor(Zhat), U{1}(:,kk), 1), U{2}(:,kk), 1), U{3}(:,kk), 1) - ...
    lamV*norm(U{1}(:,kk), 1) - lamW*norm(U{2}(:,kk),1);
end
