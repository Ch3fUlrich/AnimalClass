% function fh = daqTrainFig2(trdEval)
% fh = daqTrainFig2(trdEval)
%   
%   201115 SK

%%
startup
infile = 'd:\Steffen\scratch\DON-002865\20200917\TRD-2P\DON-002865_20200917_TRD-2P_S1-ACQ.mat';
s2pfile = 'u:\Scientific Data\RG-FD02-Data01\Steffen\Transfer\from Femtonics\DON-002865\20200917\002P-F\tif\suite2p\plane0\Fall.mat';
mescfile = 'u:\Scientific Data\RG-FD02-Data01\Steffen\Transfer\from Femtonics\DON-002865\20200917\002P-F\DON-002865_20200917_002P-F_S1-S3-ACQ.mesc';

infile = 'w:\Users\Steffen\scratch\DON-003484\20201227\TRD-2P\DON-003484_20201227_TRD-2P_S1-ACQ.mat';
infile = 'w:\Users\Steffen\scratch\DON-003484\20201227\TRD-2P\DON-003484_20201227_TRD-2P_S2-ACQ.mat';
infile = 'w:\Users\Steffen\scratch\DON-003484\20201227\TRD-2P\DON-003484_20201227_TRD-2P_S3-ACQ.mat';
s2pfile = 'w:\Users\Steffen\scratch\DON-003484\20201227\suite2p\plane0\Fall.mat';
mescfile = 'd:\Steffen\scratch\DON-003484\20201227\002P-F\DON-003484_20201227_002P-F_S1-S2-S3-ACQ.mesc'

infile = 'w:\Users\Steffen\scratch\DON-003484\20201229\TRD-2P\DON-003484_20201229_TRD-2P_S1-ACQ.mat';
infile = 'w:\Users\Steffen\scratch\DON-003484\20201229\TRD-2P\DON-003484_20201229_TRD-2P_S2-ACQ.mat';
infile = 'w:\Users\Steffen\scratch\DON-003484\20201229\TRD-2P\DON-003484_20201229_TRD-2P_S3-ACQ.mat';
s2pfile = 'w:\Users\Steffen\scratch\DON-003484\20201229\suite2p\plane0\Fall.mat';
mescfile = 'd:\Steffen\scratch\DON-003484\20201229\002P-F\DON-003484_20201229_002P-F_S1-S2-S3-ACQ.mesc'

infile = 'w:\Users\Steffen\scratch\DON-003484\20201230\TRD-2P\DON-003484_20201230_TRD-2P_S1-ACQ.mat';
infile = 'w:\Users\Steffen\scratch\DON-003484\20201230\TRD-2P\DON-003484_20201230_TRD-2P_S2-ACQ.mat';
infile = 'w:\Users\Steffen\scratch\DON-003484\20201230\TRD-2P\DON-003484_20201230_TRD-2P_S3-ACQ.mat';
s2pfile = 'w:\Users\Steffen\scratch\DON-003484\20201230\suite2p\plane0\Fall.mat';
mescfile = 'd:\Steffen\scratch\DON-003484\20201230\002P-F\DON-003484_20201230_002P-F_S1-S2-S3-ACQ.mesc'

infile = 'w:\Users\Steffen\scratch\DON-003484\20201231\TRD-2P\DON-003484_20201231_TRD-2P_S1-ACQ.mat';
infile = 'w:\Users\Steffen\scratch\DON-003484\20201231\TRD-2P\DON-003484_20201231_TRD-2P_S2-ACQ.mat';
infile = 'w:\Users\Steffen\scratch\DON-003484\20201231\TRD-2P\DON-003484_20201231_TRD-2P_S3-ACQ.mat';
s2pfile = 'w:\Users\Steffen\scratch\DON-003484\20201231\suite2p\plane0\Fall.mat';
mescfile = 'd:\Steffen\scratch\DON-003484\20201231\002P-F\DON-003484_20201231_002P-F_S1-S2-S3-ACQ.mesc'

infile = 'u:\RG Donato\Microscopy\Steffen\Transfer\from Femtonics\DON-003484\20210106\TRD-2P\DON-003484_20210106_TRD-2P_S1-ACQ.mat';

outfile = strcat(infile(1:end-4),'_eval.mat')
load(outfile);

%%
load(s2pfile);
timecourse = lowpass(F(80,:),.005)';

i = h5info(mescfile)
% attribValue = listMEScH5ObjAttribs(mescfile,'/');

unitIDs = [];
frTime = [];
nUnits = numel(i.Groups.Groups);
for iU = 1:nUnits
    unitIDs(iU) = str2num(i.Groups.Groups(iU).Name(end));
end

for iU = 1:nUnits
    [~,attrList] = listMEScH5ObjAttribs(mescfile,strcat('/MSession_0/MUnit_',num2str(unitIDs(iU))));
    clc;
    [r,c] = find(strcmp(attrList,'ZAxisConversionConversionLinearScale'));
    frTime(iU,1) = attrList{r,c+1};    
end
frTime
sf = 1000/frTime(1)

tt = [0:numel(timecourse)-1]./sf;

%%
startupfigs 1;
fh = [];

%%
tt = trdEval.time;
dist = trdEval.distance_corr./100;
spd = trdEval.speed_raw;
pos = trdEval.position_corr;

%% new analysis
[p,b] = histc(pos,1:180);

idx = [[1;(find(diff(pos(1:end-1)))+1)],[find(diff(pos));length(tt)]];
idx(:,3) = diff(idx,1,2)+1;
idx(end,[2 3]) = idx(end,[2 3])-1;


cmpb = 2; % [cm]
cmpb = trdEval.defaults.treadmill.beltLen/99; % N=100 bins
posBins = 0:cmpb:trdEval.defaults.treadmill.beltLen;

[~,binnedPos] = histc(trdEval.position_corr,posBins);

tic
runMaps = zeros(numel(posBins),numel(trdEval.indices.lapOnset),10);

stops = zeros(size(trdEval.position_raw));
stops(trdEval.indices.runOffset) = 1;
er = 0;
disp(num2str(numel(trdEval.indices.lapOnset)-1))
for iL = 1:numel(trdEval.indices.lapOnset)-1
    disp(num2str(iL))
    here = trdEval.indices.lapOnset(iL):trdEval.indices.lapOnset(iL+1)-1;
    p = trdEval.position_corr(here);
    [n,i] = histc(p,posBins);
    p = trdEval.position_corr(here);
    s = trdEval.speed_raw(here);
    r = stops(here);
    try
        for iB = 1:numel(posBins)
            idx = find(i == iB);
            if ~isempty(idx)
                runMaps(iB,iL,1) = nanmean(p(idx));
                runMaps(iB,iL,2) = nanmean(s(idx));
                runMaps(iB,iL,3) = nanmean(r(idx));
            end
        end
    catch
        disp(num2str(er))
        er = er+1;
    end
end
toc
runMaps(:,:,3) = runMaps(:,:,3)./runMaps(:,:,3);
runMaps(isnan(runMaps)) = 0;

%%
fh = figure;

% dist
ah1 = ax([2 12 10 3]);
plot(ah1,tt,dist,'k-')
xlabel(ah1,'Time (s)')
ylabel(ah1,'Distance (m)')
xlim(ah1,tt([1 end]))
ylim(ah1,[0 round2base(max(dist),10,'ceil')])
title(ah1,'Running distance')

% speed
ah2 = ax([2 7 10 3]);
plot(ah2,tt,spd,'k-')
xlabel(ah2,'Time (s)')
ylabel(ah2,'Speed (cm/s)')
xlim(ah2,tt([1 end]))
ylim(ah2,[0 round2base(max(spd),10,'ceil')])
title(ah2,'Running speed')

% pos
ah3 = ax([2 2 10 3]);
plot(ah3,tt,pos,'k-')
xlabel(ah3,'Time (s)')
ylabel(ah3,'Position (cm)')
xlim(ah3,tt([1 end]))
ylim(ah3,[0 trdEval.defaults.treadmill.beltLen])
title(ah3,'Track position')

% spd/lap
ax([15 7 4 3],'isc');
imagesc(posBins,1:size(runMaps,2)-1,squeeze(runMaps(:,1:end-1,2))',[0 round2base(max(spd),10,'ceil')])
ah4 = gca;
xlabel(ah4,'Position (cm)')
ylabel(ah4,'Lap #')
cbh1 = drawcb(ah4,[.5 .5]);
ylabel(cbh1,'Speed (cm/s)')
colormap(parula(64))
freezeColors(ah4);
title(ah4,'Speed map')

ah5 = ax([22 7 4 3]);
for iL = 1:size(runMaps,2)-1
    plot(ah5,posBins,runMaps(:,iL,2),'-','Color',[1 1 1]*.8)
end
mSpd = nanmean(runMaps(:,1:end-1,2),2);
h = plot(ah5,posBins,mSpd,'k-');
xlim(ah5,posBins([1 end]))
ylim(ah5,[0 round2base(max(spd),10,'ceil')])
xlabel(ah5,'Position (cm)')
ylabel(ah5,'Speed (cm/s)')
title(ah5,'Lap speed')
lh = legend(h,'Mean');
set(lh,'Box','off','Location','NorthEast')

% stop pos
ax([15 2 4 3],'isc');
imagesc(posBins,1:size(runMaps,2)-1,squeeze(runMaps(:,1:end-1,3))',[0 1])
ah6 = gca;
xlabel(ah6,'Position (cm)')
ylabel(ah6,'Lap #')
colormap(cat(1,[1 1 1],[1 0 0]))
freezeColors(ah6);
title(ah6,'Stops (>1 sec.)')

% run/stat bouts
runOn = trdEval.indices.runOnset;
runOff = trdEval.indices.runOffset;
sf = trdEval.info.samplFreq;

tBin = 5;
tBins = 0:tBin:120;
n1 = hist(diff([runOn runOff],1,2)./sf,tBins);
n2 = hist(diff([runOff(1:end-1) runOn(2:end)],1,2)./sf,tBins);

ah7 = ax([22 2 4 3]);
bh2 = bar(ah7,tBins+tBin/2,n2,1,'FaceColor','r','EdgeColor','none','FaceAlpha',.5);
bh1 = bar(ah7,tBins+tBin/2,n1,1,'FaceColor','b','EdgeColor','none','FaceAlpha',.5);
xlim(ah7,tBins([1 end]))
xlabel(ah7,'Duration (s)')
ylabel(ah7,'N')
title(ah7,'Bout histogram')
lh = legend('Stops','Runs');
set(lh,'Box','off','Location','NorthEast')

ah8 = ax([14.5 12 .5 3]);
bar(ah8,1,trdEval.meters,.25,'FaceColor','k','EdgeColor','none','FaceAlpha',.5)
set(ah8,'XTick',[])
xlim(ah8,[.5 1.5])
ylim(ah8,[0 round2base(trdEval.meters,10,'ceil')])
ylabel(ah8,'Distance (m)')
title(ah8,'Overview')

ah9 = ax([16 12 .5 3]);
bar(ah9,1,trdEval.laps,.25,'FaceColor','k','EdgeColor','none','FaceAlpha',.5)
set(ah9,'XTick',[])
xlim(ah9,[.5 1.5])
ylim(ah9,[0 round2base(trdEval.laps,10,'ceil')])
ylabel(ah9,'# Laps')

ah10 = ax([17.5 12 .5 3]);
mpm = trdEval.meters/trdEval.duration;
bar(ah10,1,mpm,.25,'FaceColor','k','EdgeColor','none','FaceAlpha',.5)
set(ah10,'XTick',[])
xlim(ah10,[.5 1.5])
ylim(ah10,[0 round2base(mpm,1,'ceil')])
ylabel(ah10,'Meter per min.')

ah11 = ax([19 12 .5 3]);
bar(ah11,1,trdEval.speed.runRatio,.25,'FaceColor','b','EdgeColor','none','FaceAlpha',.5)
set(ah11,'XTick',[])
xlim(ah11,[.5 1.5])
ylim(ah11,[0 1])
ylabel(ah11,'Run ratio')

ah12 = ax([22 12 4 3]);
plot(ah12,trdEval.mins.distance,'s-','Color',[1 1 1]*.8,'MarkerFaceColor','k','MarkerEdgeColor','none')
xlim(ah12,[0 trdEval.duration])
ylim(ah12,[0 max(trdEval.mins.distance)])
xlabel(ah12,'Time (min)')
ylabel(ah12,'Distance (m)')
title(ah12,'Distance per min.')

ah13 = ax([29 12 4 3]);
plot(ah13,trdEval.mins.laps(1:end-1),'s-','Color',[1 1 1]*.8,'MarkerFaceColor','k','MarkerEdgeColor','none')
xlim(ah13,[0 trdEval.duration])
ylim(ah13,[0 max(trdEval.mins.laps(1:end-1))])
xlabel(ah13,'Time (min)')
ylabel(ah13,'N')
title(ah13,'Laps per min.')

ah14 = ax([29 7 4 3]);
plot(ah14,trdEval.mins.avgSpeed,'s-','Color',[1 1 1]*.8,'MarkerFaceColor','b','MarkerEdgeColor','none')
plot(ah14,trdEval.mins.topSpeed,'s-','Color',[1 1 1]*.8,'MarkerFaceColor','c','MarkerEdgeColor','none')
xlim(ah14,[0 trdEval.duration])
ylim(ah14,[0 round2base(max(trdEval.mins.topSpeed),10,'ceil')])
xlabel(ah14,'Time (min)')
ylabel(ah14,'Speed (cm/s)')
title(ah14,'Speed per min.')
lh = legend('Mean','Peak');
set(lh,'Box','off','Location','NorthEast')

% th = fliplr(linspace(.5*pi,2.5*pi,size(trdEval.speed.avgBout,1)));
% ph = polar(th',trdEval.speed.avgBout(:,1),'bs-');
ah15 = ax([29 2 4 3]);
plot(ah15,trdEval.speed.avgBout(:,1),'s-','Color',[1 1 1]*.8,'MarkerFaceColor','b','MarkerEdgeColor','none')
plot(ah15,trdEval.speed.peakBout,'s-','Color',[1 1 1]*.8,'MarkerFaceColor','c','MarkerEdgeColor','none')
xlim(ah15,[0 size(trdEval.speed.avgBout,1)+1])
ylim(ah15,[0 round2base(max(trdEval.speed.peakBout),10,'ceil')])
xlabel(ah15,'Bout #')
ylabel(ah15,'Speed (cm/s)')
title(ah15,'Speed per bout')
lh = legend('Mean','Peak');
set(lh,'Box','off','Location','NorthEast')

colormap(parula(64))

[~,n,~] = fileparts(infile);
suptitle(fh,n);

ticks
fs(10)
fig

%%
saveFigAsPDF(fh,n);
close(fh);


