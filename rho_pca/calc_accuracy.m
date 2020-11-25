function acc = calc_accuracy(Y, Yhat, varargin)
%==========================================================================
% This function calculates the classification accuracy, as follows
%       acc = (# correct predictions)/(total # predictions)
%
% INPUT:
%   1. 'Y': (n x g) binary matrix of true values
%   2. 'Yhat': (n x g) binary matrix of predicted values
%   3. 'verbose' (default false): Boolean value indicating to print all
%   messages in code
%
% OUTPUT:
%   1. 'acc': Accuracy of prediction
%==========================================================================

%% Load and verify parameters
% Specify default parameters and their expected values
default_verbose = false;
% Parse input parameters
p = inputParser;
p.CaseSensitive = true;
addRequired(p, 'Y');
addRequired(p, 'Yhat');
addParameter(p, 'verbose', default_verbose, @islogical);
parse(p, Y, Yhat, varargin{:});
verbose = p.Results.verbose;
% Check the dimensionality of Y and Yhat
n = size(Y,1);
g = size(Y,2);
if (n ~= size(Yhat,1))||(g ~= size(Yhat,2))
    error('Y and Yhat must have the same dimensions.\n'); 
end

%% Calculate accuracy
Y = logical(Y);
Yhat = logical(Yhat);
num_correct_pred = sum(Y(:) == Yhat(:));
acc = num_correct_pred/(n*g);
if verbose; fprintf('The prediction accuracy is %.2f.\n', acc); end
end




