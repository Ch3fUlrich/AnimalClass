function shuffledData = shuffleData(indata,nShuff,shuffMeth)
% SHUFFLE (ADVANCING SHIFT) INPUT DATA FOR STATISTICAL COMPARE
%   shuffledData = shuffleData(indata,nShuff,shuffMeth)
%   
%   <indata>    - 2D: nSamples x nVars (e.g. traces, trials, cells, ...)
%   <nShuff>    - default=100
%   <shuffMeth> - 'random' or 'regular' shift
%   
%   SK 120916 V2, n fix
%   SK 210624 V1

if nargin < 3
    shuffMeth = 'rand';
end
if nargin < 2
    nShuff = 100;
end

[nSamples,nVars] = size(indata);

shuffledData = zeros(nSamples,nVars,nShuff);

switch shuffMeth
    case {'rand','random','rnd'}
        shifts = randi([1,nSamples],1,nShuff);
    case {'reg','regular','fix','fixed'}
        n = nSamples-floor(nSamples/nShuff);
        shifts = floor(n/nShuff:n/nShuff:n);
    otherwise
        error('this shuffling is not implemented')
end

wb = waitbar(0);
for iShuff = 1:nShuff
    waitbar(iShuff/nShuff,wb);
    shuffledData(:,:,iShuff) = circshift(indata,[-shifts(iShuff) 0]);
end
close(wb);

shuffledData = squeeze(shuffledData);
