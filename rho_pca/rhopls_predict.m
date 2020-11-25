function [Yhat, RT] = rhopls_predict(Xtest, U, fmod, varargin)
%==========================================================================
% This function performs rho-PLS, and returns i.) reduced form of the
% original data X, and ii.) Fits a model using the reduced data X and
% Y. 
%
% INPUT:
%   1. 'Xtest': Test data as a (n x p x q x r)-dim. tensor
%   2. 'U': (4 x 1) cell array of (n x K) factors, where n is the 
%       dimensionaltiy of the respective mode
%   3. 'fmod': model object produced by the rhopls_fit function
%   4. 'verbose' (default true): Optional, boolean value indicating to 
%       print all messages in code
%
% OUTPUT:
%   1. 'Yhat': estimated (ntest x l) response matrix, where l is the number
%       of response variables
%   2. 'RT': Run time (seconds)
%
% Dependencies: Tensor Toolbox package
%==========================================================================

%% Load and verify parameters
% Specify default parameters and their expected values
default_verbose = true;
p = inputParser;
p.CaseSensitive = true;
addRequired(p, 'Xtest', @(ii) isa(ii,'double')||isa(ii,'tensor'));
addRequired(p, 'U');
addRequired(p, 'fmod');
addParameter(p, 'verbose', default_verbose, @islogical);
parse(p, Xtest, U, fmod, varargin{:});
verbose = p.Results.verbose;
% Check dimensionality of covariance Xtest
Xtest = tensor(squeeze(Xtest));         
nD = length(size(Xtest));
if nD ~= 4; error('Data must be a 4-dimensional tensor');  end
vIdx = 1;
wIdx = 2;
tIdx = 3;

%% Predict rho-PLS
if verbose; fprintf('Make rho-PLS prediction.\n'); end

tic;
% Reduce testing data
K = size(U{1}, 2);
n = size(Xtest, 1);
Xtilde = nan(n, K);
for kk = 1:K
    Xtilde(:,kk) = double(ttv(ttv(Xtest, U{3}(:,kk), 4), U{2}(:,kk), 3))*U{1}(:,kk);
end

% Predict Yhat
Yhat = predict(fmod, Xtilde);
RT = toc;
end




