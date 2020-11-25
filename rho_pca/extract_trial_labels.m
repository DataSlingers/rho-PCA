function [y] = extract_trial_labels(integer_labs, lab)
% ========================================================================
% This fucntion assigns binary labels to FC data
%
% Integer label keys:
% 1: rain, clear audio, clear visual
% 2: rock, clear audio, clear visual
% 3: rain, noisy audio , clear visual
% 4: rock, noisy audio, clear visual
% 5: rain, clear audio, noisy visual
% 6: rock, clear audio, noisy visual
% 7: rain, noisy audio, noisy visual
% 8: rock, noisy audio, noisy visual
%
% The potnetial label labels are,
% 1. 'visual' - 1 for clear visual, 0 otherwise
% 2. 'audio' - 1 for clear audio, 0 otherwise
% 3. 'word' - 1 for rock, 0 for rain
%
% INPUT:
% 1. integer_labs - vector of integer labels
% 2. lab - label to create (e.g., 'visual', 'audio', 'word')
% =========================================================================

% Check parameters
if ~any(strcmp(lab, {'visual', 'audio', 'word'}))
    error("The only models that may be specified are 'visual', 'audio', or 'word'.\n");
end
n = length(integer_labs);
assign_ints = {1:4, [1,2,5,6], [2,4,6,8]};

% Create labels
y = zeros(n,1);
for ii = 1:n
    if strcmp(lab,'visual')&&any(assign_ints{1}==integer_labs(ii))
        y(ii) = 1;
    elseif strcmp(lab,'audio')&&any(assign_ints{2}==integer_labs(ii))
        y(ii) = 1;
    elseif strcmp(lab,'word')&&any(assign_ints{3}==integer_labs(ii))
        y(ii) = 1;
    end
%     else
%         fprintf("CANNOT RECOGNIZE LABEL!!!\n");
%         continue;
    %end
end
end
