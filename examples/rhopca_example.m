% ========================================================================
%                   rhoPCA Example
%                   (Takes approx. 4 minutes to run)
%
% Author:       Kelly Geyer
% Date:         November 24, 2020
% Contact:      klgeyer@bu.edu
% Development:  Matlab R2019a
%
% Description:
% This file provides an example of tuning and fitting rho-PCA.
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
% Candidate parameters for hyperparameter tuning
candidate_lamV = [10.^(-1:0),.5];                   
candidate_lamW = [10.^(-1:0),.5];                       
candidate_alphaW = [10.^(-1:0),.5];                   
candidate_alphaT = [10.^(-1:0),.5];                    
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
X = X(1:20,1:20,1:20,1:20);


%% Find the optimal hyperparameters for rho-PCA
% This function implements nested BIC tuning described by Algorithm 3 in 
% the paper.
tune = tune_rhopca(X, 'K', K, ...
    'candidate_lamV', candidate_lamV, ...
    'candidate_lamW', candidate_lamW, ...
    'candidate_alphaW', candidate_alphaW, ...
    'candidate_alphaT', candidate_alphaT);


%% Perform rho-PCA tensor decomposition on data
% This function performs rho-PCA to find a set of interpretable factors
% that capture the variance in the data. It implements Algorithm 1 from the
% paper.
[U, ~, ~, ~] = rhopca(X, 'K', K, ...
    'lamV', tune.optimal_params.lamV, ...
    'lamW', tune.optimal_params.lamW, ...
    'alphaW', tune.optimal_params.alphaW, ...
    'alphaT', tune.optimal_params.alphaT);


%% Visualize Components
% Refer to Table 1, as well as figures, from the original paper for ideas
% of potential visualization with the rho-PCA factors. In this section, we
% plot the Cumlative Amount of Variance Explained (CAVE).
cave_vals = cave(U, X, 'k', K);
% Plot CAVE values
plot(1:K, cave_vals, '-*');
ylabel('CAVE');
xlabel('Component (k)');
title('Cumlative Amount of Variance Explained')
shg
