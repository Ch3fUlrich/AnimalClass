function spddefs = speedDefs
% speedDefs loads default params for speed analyses
%   
%   200526 SK
%   200511 SK


%% defs
runThr = 2; % [cm/s], min peak speed in running bout
minDur = 1; % [s], min running bout duration
offDel = 1; % [s] offset period after stopping; e.g. for Ca signal decay
pauseThr = 1; % [s], min pause duration for stationarity 

runBins = [minDur 5:5:60]; % [s], for histo
pauseBins = [pauseThr 5:5:60]; % [s], for histo
spdBins = [runThr 5:5:60]; % [s], for histo 
avgBins = [runThr:2:40]; % [s], for histo
pkBins = [5:5:60]; % [s], for histo

%%
spddefs.runThr = runThr;
spddefs.minDur = minDur;
spddefs.offDelay = offDel;
spddefs.pauseThr = pauseThr;

spddefs.histo.run = runBins;
spddefs.histo.pause = pauseBins;
spddefs.histo.speed = spdBins;
spddefs.histo.avg = avgBins;
spddefs.histo.peak = pkBins;

disp('Default parameters for SPEED loaded')
