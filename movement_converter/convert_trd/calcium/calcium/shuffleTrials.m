function [shuffledTrials,idx] = shuffleTrials(indata,shuffMeth)
% SHUFFLES INPUT TRIALS
%   shuffledTrials = shuffleTrials(indata,shuffMeth)
%   
%   <indata>    - 2D: nSamples x nTrials 
%   <shuffMeth> - 'odd', 'even', 'half', 'rand', ...
%   
%   SK 210626 V1

[nSamples,nTrials] = size(indata);

shuffledTirals = zeros(size(indata));

switch shuffMeth
    case {'odd','even'}
        idx = [1:2:nSamples 2:2:nSamples];
    case {'half','50'}
        idx = [1:ceil(nSamples/2) ceil(nSamples/2)+1:nSamples];
    case {'rand','rnd','random','perm','permute','randperm'}
        idx = randperm(nTrials);
    otherwise
        error('this shuffling is not implemented')
end

shuffledTrials = indata(:,idx);
