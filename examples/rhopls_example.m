% ========================================================================
%                   rhoPLS Example
%                   (Takes approx. 4 minutes to run)
%
% Author:       Kelly Geyer
% Date:         November 24, 2020
% Contact:      klgeyer@bu.edu
% Development:  Matlab R2019a
%
% Description:
% This file provides an example of tuning and fitting rho-PLS.
%==========================================================================

%% Set parameters
% Set data file name
data_fn = "/Users/kelly/GoogleDriveBU/rhoPCA/data/FC_data/trial_center_scale/YAH/trial_center_scale_YAH.h5";
% Path to Tensor Toolbox
ttb_path = "/Users/kelly/Documents/gitlab/tensor_toolbox";
% Path to rho-pca code
rpca_path = "/Users/kelly/Documents/github/rho-PCA/rho_pca";
% Number of pseudo-indepentdent factors to generate.
K = 3;
% Number of cross-validation folds
nFolds = 5;
% Candidate parameters for hyperparameter tuning
candidate_lamV = [0.1, .5];                   
candidate_lamW = [0.1, .5];                         
candidate_alphaW = [0.1, .5];                    
candidate_alphaT = [0.1, .5];                      
% Set seed value for reproducabiltiy
seed.value = 123;


%% Configure the script
% Set seed for reproducabiltiy
rng(seed.value);
% Load dependencies
addpath(ttb_path);              
addpath(rpca_path);   
savepath;   


%% Load data
% 1. Load ECoG tensor. We use a subset of this data for ease of
% demonstration.
fprintf('Loading covariates: %s.\n', data_fn);
X = h5read(data_fn, '/X');
X = X(1:50, 1:20, 1:20, 1:20);
% 2. Load trial labels
fprintf('Loading response: %s\n', data_fn);
trial_labels = h5read(data_fn, '/labels');
y = extract_trial_labels(trial_labels, 'visual');
y = y(1:50);


%% Partition data into training and testing sets
% Index 1 will serve as the testing set
% Indexes 2-nFolds will serve as training sets
partIdx = cvpartition(y, 'KFold', nFolds);
parts = cell(nFolds,1); 
for ii = 1:nFolds; parts{ii} = partIdx.test(ii); end


%% Find the optimal hyperparameters for rho-PLS
% This function implements nested cross-validation described by Algorithm 4 
% in the paper.
tune = tune_rhopls(X, y, parts, 'K', K, ...
    'candidate_lamV', candidate_lamV, ...
    'candidate_lamW', candidate_lamW, ...
    'candidate_alphaW', candidate_alphaW, ...
    'candidate_alphaT', candidate_alphaT);


%% Perform rho-PLS tensor decomposition on data
% This function performs rho-PLS to find a set of interpretable factors
% that capture the variance in the data. It implements Algorithm 2 from the
% paper.
[U, fmod, out] = rhopls_fit(X, y, 'K', K, ...
    'lamV', tune.optimal_params.lamV, ...
    'lamW', tune.optimal_params.lamW, ...
    'alphaW', tune.optimal_params.alphaW, ...
    'alphaT', tune.optimal_params.alphaT);


%% Calculate accuracy of rho-PLS + LDA
% In this section we fit rho-PLS + LDA to the data, and calculate the
% accuracy. Since we have taken a very small subset of data, our accuracy
% will be low in this example.
[Yhat, RT] = rhopls_predict(X, U, fmod);
acc = calc_accuracy(y, Yhat);
fprintf("Our accuracy is %.2f\n", acc);

