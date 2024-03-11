function outdata = daqTrainEval(infile,overwriteFlag)
% EXTRACTS DATA FROM TREADMILL BEHAVIOR
%   outdata = daqTrainEval(infile,overwriteFlag)
%   
%   <infile> full file name; to be replaced by experiment identifier, e.g. 'DON-000123\YYYYMMDD'
%   <overwriteFlag> '0' or '1'
%   <outdata> structure with results from training eval 
%
%   See also DAQTRAINDEFS, DAQTRAINREAD, DAQTRAINFIG, RUNEVAL
%   
%   200607 SK V3, some more updates and fixes
%   200527 SK V2, some updates
%   200328 SK V1

if nargin < 2
    overwriteFlag = 0;
end

%%
% startup

% overwriteFlag = 1;

% infile = 'd:\Steffen\scratch\DON-002865\20200917\TRD-2P\DON-002865_20200917_TRD-2P_S3-ACQ.mat'
% infile = 'd:\Steffen\scratch\DON-002865\20200917\TRD-2P\DON-002865_20200917_TRD-2P_S1-ACQ.mat'

% infile = 'u:\Scientific Data\RG-FD02-Data01\Steffen\Transfer\from Femtonics\DON-002865\20201109\TRD-2P\DON-002865_20201109_TRD-2P_S1-ACQ.mat'
% infile = 'u:\Scientific Data\RG-FD02-Data01\Steffen\Transfer\from Femtonics\DON-002865\20201109\TRD-2P\DON-002865_20201109_TRD-2P_S2-ACQ.mat'
% infile = 'u:\Scientific Data\RG-FD02-Data01\Steffen\Transfer\from Femtonics\DON-002865\20201109\TRD-2P\DON-002865_20201109_TRD-2P_S3-ACQ.mat'

% infile = 'u:\RG Donato\Microscopy\Steffen\Transfer\from Femtonics\DON-003484\20210106\TRD-2P\DON-003484_20210106_TRD-2P_S1-ACQ.mat';



% % % infile = 'f:\Users\Steffen\scratch\DON-004366\20210225\TRD-2P\DON-004366_20210225_TRD-2P_S1-ACQ.mat' % <== this file
% infile = 'f:\Users\Steffen\scratch\DON-004366\20210228\TRD-2P\DON-004366_20210228_TRD-2P_S1-ACQ.mat'
% infile = 'f:\Users\Steffen\scratch\DON-003484\20201216\TRD-2P\DON-003484_20201216_TRD-2P_S1-ACQ.mat'

% if nargin < 2
%     saveflag = 1;
%     disp('Force-save engaged')
% end+
% if nargin < 1
%     expt = input('Specify experiment: ','s'); % pass as e.g. 'DON-000123\YYYYMMDD' 
% end
% 
% dirs = getDirs(expt);

outdata = [];
outfile = '';
outfile = strcat(infile(1:end-4),'_eval.mat')

%% load data
if exist(outfile,'file') && overwriteFlag == 0
    load(outfile);
    outdata = trdEval;
    return;
else
    load(infile);
    % load(dirs...) FIXME
end

%% defs
dtdefs = daqTrainDefs;

% overwrites defaults 
if isfield(metadata,'trackLength')
    dtdefs.beltLen = metadata.trackLength;
    disp('Belt length overwritten by METADATA')
end
if isfield(metadata,'samplFreq') 
    dtdefs.sf = metadata.samplFreq;
    disp('SF overwritten by METADATA')
end
disp(['Belt length is ',num2str(dtdefs.beltLen),' cm'])
disp(['SF is ',num2str(dtdefs.sf),' Hz'])

%% extract data
tic
s = strfind(infile,'TRD-TR');
if ~isempty(s)
    data = trainingdata;
end

s = strfind(infile,'TRD-2P');
if ~isempty(s)
    data = trainingdata;
end

nos = length(data); % no. sampling points

tt = ([0:nos-1]./dtdefs.sf)'; % time axis

encChA = data(:,1); % enc
encChB = data(:,2); % enc
lapCh = data(:,3); % lap

% % raw fig
% figure;
% subplot(4,1,1)
% ah1 = gca; hold on
% plot(ah1,tt,encChA,'k')
% subplot(4,1,2)
% ah2 = gca; hold on
% plot(ah2,tt,encChB,'k')
% subplot(4,1,3)
% ah3 = gca; hold on
% plot(ah3,tt,lapCh,'k')
% try
%     subplot(4,1,4)
%     ah4 = gca; hold on
%     plot(ah4,tt,data(:,4),'k')
% end
% linkaxes(gaa,'x');

% detect thr crossings
encChA(encChA < dtdefs.thr.TTL) = 0;
encChA(encChA >= dtdefs.thr.TTL) = 1;
encChB(encChB < dtdefs.thr.TTL) = 0;
encChB(encChB >= dtdefs.thr.TTL) = 1;

% lapCh = lowpass(lapCh,dtdefs.misc.lowpass_lap); % obsolete since 200923
% dtdefs.thr.lap = min(lapCh)+(range(lapCh)*dtdefs.thr.lap); % same
lapCh = lowpass(lapCh,dtdefs.misc.lowpass_lap); % added back in, seems to fix fast transients
lapThr = dtdefs.thr.lap;
lapThr = min(lapCh)+range(lapCh)/2;
disp('warning: threshold for lap detection adapted to input range')

lapCh(lapCh < lapThr) = 0;
lapCh(lapCh >= lapThr) = 1;

tmp = diff([1;lapCh]);
idxLapLo = find(tmp == -1)+1;
idxLapHi = find(tmp == 1);
% idxLapLo = idxLapLo(1:numel(idxLapHi)); % dirty fix ?
% idxLapOn = idxLapLo; % obsolete since 200923
idxLapOn = idxLapHi; 

% % raw fig
% plot(ah3,tt(idxLapOn),lapThr,'b>')

% rotation
% tic
% rota = encoder(encChA',encChB'); % slow
% toc
rota = quadenc(encChA,encChB);
if dtdefs.misc.invert
    rota = -rota;
    disp('rotation inverted')
end
% rota = rota';

% rad dist
% figure;
% subplot(4,1,1)
% ah1 = gca; hold on
% plot(ah1,tt,rota,'k')
% ylim(ah1,[-2 2])
% subplot(4,1,2:3)
% ah2 = gca; hold on
% plot(ah2,tt,cumsum(rota),'k')
% linkaxes(gaa,'x');

idxRota = find(rota);

fwdIncr = zeros(size(rota));
csRota = cumsum(rota); % raw move
tic
[~,idx,~] = unique(csRota(csRota > 0),'first');
try
    idx = idx+numel(find(csRota == 0));
catch
    % do nothing
end
fwdIncr(idx) = 1; % net move


% for ii = 1:numel(this)
%     if ~mod(ii,1000)
%         disp('o')
%     end
%     idx = find(csRota == this(ii),1,'first');
%     fwdIncr(idx) = 1;
% end
% % idx = find(csRota(2:end) > csRota(1:end-1));
% for ii = 2:numel(csRota)
%     if csRota(ii) > max(csRota(1:ii-1))
%         fwdIncr(ii) = 1;
%     end
% end
toc
% plot(ah1,tt(idx),rota(idx),'ro')
% 
% plot(ah2,tt,cumsum(fwdIncr),'r-')
tmp = [];
for iL = 1:numel(idxLapOn)-1
%     plot(ah1,tt([1 1]*(idxLapOn(iL))),[-2 2],'-','Color',[1 1 1]*.75)
%     plot(ah2,tt([1 1]*(idxLapOn(iL))),minimax(csRota),'-','Color',[1 1 1]*.75)
    tmp(iL,1) = numel(find(fwdIncr(idxLapOn(iL):idxLapOn(iL+1))));
end

pplIdeal = dtdefs.beltLen/dtdefs.wheel.dpp;
idxLapOnReal = [];
islapTmp = zeros(size(idxLapOn));
islapTmp(1) = 1;
for iL = 1:numel(tmp)
    this = tmp(iL);
    if this < pplIdeal*0.9
        islapTmp(iL+1) = -1;
    elseif this > pplIdeal*1.1
        islapTmp(iL+1) = -2;
    else
        islapTmp(iL+1) = 1;
    end
end
idxLapOnReal = idxLapOn(find(islapTmp == 1));

tmp = [];
for iL = 1:numel(idxLapOnReal)-1
    tmp(iL,1) = numel(find(fwdIncr(idxLapOnReal(iL):idxLapOnReal(iL+1))));
end

figure
plot(tt,data(:,3),'k-')
hold on
plot(tt(idxLapOnReal),4,'ro')

% xlim([1314 1328])

idxLapOn = idxLapOnReal;
rota = fwdIncr;

% % laps ?
% figure;
% ah = gca; hold on
% cmp = parula(numel(tmp));
% tmp_pos = zeros(size(fwdIncr));
% for ii = 1:numel(tmp)
%     pw = dtdefs.beltLen/tmp(ii);
%     here = idxLapOnReal(ii):idxLapOnReal(ii+1)-1;
%     idx = find(fwdIncr(here));
%     this = ones(size(idx))*pw;
%     tmp_pos(here(idx)) = this;
%     this = cumsum(this);
% %     plot(ah,[0:numel(this)-1]./numel(this),this+ii*10,'.-','Color',cmp(ii,:))
%     plot(ah,this+ii*10,'.-','Color',cmp(ii,:))
% end
    
% % tmp_pos = zeros(size(rota));
% % for ii = 1:numel(tmp)
% %     pw = dtdefs.beltLen/tmp(ii);
% %     here = idxLapOnReal(ii):idxLapOnReal(ii+1)-1;
% %     idx = find(fwdIncr(here));
% %     tmp_pos(here(idx)) = pw;
% % end
% % tmp_pos = cumsum(tmp_pos);
% % tmp_pos_cs = mod(tmp_pos,dtdefs.beltLen);
% % figure
% % plot(tmp_pos)
% % figure
% % plot(tmp_pos_cs)
% % 
% % figure;
% % ah = gca; hold on
% % cmp = parula(numel(tmp));
% % for ii = 1:numel(tmp)
% %     idx = find(fwdIncr(idxLapOnReal(ii):idxLapOnReal(ii+1)-1));
% %     this = tmp_pos_cs(idx);
% %     idx2 = find(this);
% %     plot(ah,[0:numel(this(idx2))-1]./numel(this(idx2)),this(idx2),'.-','Color',cmp(ii,:))
% % %     plot(ah,this+ii*180,'.-','Color',cmp(ii,:))
% %     drawnow
% %     pause
% % end


    % actual no. of pulses per lap
    ppl = nan(numel(idxLapOn),1);
    for ii = 1:numel(idxLapOn)-1
        ppl(ii,1) = sum(rota(idxLapOn(ii):idxLapOn(ii+1)-1));
    end
    avgppl = floor(nanmean(ppl)); % avg.

    islap = zeros(numel(ppl),1);
    islap(find(ppl >= floor(dtdefs.beltLen/(dtdefs.wheel.diam*pi)*dtdefs.wheel.ppr*(1-dtdefs.thr.encoder)))) = 1; % more ppl than cirterion
    idxLapOn = idxLapOn(find(islap)); % correct, not
    ppl = ppl(find(islap)); % correct, not
    
    idxLapOff = nan(size(idxLapOn));
    idxLapOff(1:end-1) = idxLapOn(2:end)-1;

        % running lap idx
        lapIdx = zeros(size(lapCh));
        cc = 1;
        for ii = 1:numel(idxLapOn)-1
            here = idxLapOn(ii):idxLapOn(ii+1)-1;
            lapIdx(here) = cc;
            cc = cc+1;
        end
        lapIdx(idxLapOn(end):end) = cc; 

    % dist. per pulse per lap - for correcting dist / pos
    dpp_lap = nan(size(idxLapOn));
    for ii = 1:numel(dpp_lap)
        try
            dpp_lap(ii,1) = dtdefs.beltLen/ppl(ii);
        catch
            dpp_lap(ii,1) = dtdefs.wheel.dpp;
            disp(['default lap dpp for lap #' num2str(ii)])
        end
    end

toc

figure;
hold on
plot(1:numel(ppl),ppl,'k.-')
xlim([0 numel(ppl)+1])
ylim([0 5000])

figure;
hold on
plot(tt,data(:,3)./5,'-','Color',[1 1 1]*.5)
plot(tt,lapCh,'-','Color',[1 1 1]*.8)
plot(tt(idxLapHi),1,'ko')
plot(tt(setdiff(idxLapHi,idxLapOn)),2,'r|')
plot(tt(idxLapOn),1.5,'k|')
ylim([-10 10])
% plot(tt,tmp,'-b')
% plot(tt,tmp,'-r')

%% tmp check - this does some bullshit
% % % % dtLap = diff(lowpass(-data(:,3),dtdefs.misc.lowpass_lap));
% % % % sdLap = std(dtLap);
% % % % tmp = zeros(size(dtLap));
% % % % tmp(dtLap >= sdLap*dtdefs.misc.sd_lap) = 1;
% % % % 
% % % % % figure
% % % % % ah1 = gca;
% % % % % plot(ah1,tt(1:end-1),dtLap,'k-')
% % % % % hold on
% % % % % plot(ah1,tt([1 end-1]),[1 1]*sdLap*dtdefs.misc.sd_lap,'b-')
% % % % % plot(ah1,tt([1 end-1]),[1 1]*sdLap,'k-')
% % % % % plot(ah1,tt(1:end-1),tmp.*max(dtLap),'b-')
% % % % 
% % % % tmp = diff(tmp);
% % % % idxLapOn = find(tmp == 1)+1;
% % % % idxLapOff = nan(size(idxLapOn));
% % % % idxLapOff(1:end-1) = idxLapOn(2:end)-1;
% % % % 
% % % % clear ppl_tmp
% % % % for iL = 1:numel(idxLapOn)-1
% % % % %     plot(ah1,tt(idxLapOn(iL)),sdLap*dtdefs.misc.sd_lap,'bo')
% % % %     if iL < numel(idxLapOn)
% % % %         ppl_tmp(iL,1) = numel(find(rota(idxLapOn(iL):idxLapOn(iL+1)-1)));
% % % %     end
% % % % end
% % % % avgppl = mean(ppl_tmp);
% % % % avgppl = round(dtdefs.beltLen./(dtdefs.wheel.diam*pi)*dtdefs.wheel.ppr);
% % % % 
% % % % outliers = find(ppl_tmp > avgppl*1.1 | ppl_tmp < avgppl*.9);
% % % % 
% % % % % if ~isempty(outliers)
% % % % %     plot(ah1,tt(idxLapOn(outliers)),sdLap*dtdefs.misc.sd_lap,'rx')
% % % % % end
% % % % 
% % % % dist_raw = cumsum(rota.*dtdefs.wheel.dpp)+dtdefs.wheel.dpp;
% % % % tmp = dist_raw+diff([dist_raw(idxLapOn(1)) dtdefs.beltLen]);
% % % % tmp = dist_raw+diff([dist_raw(idxLapOn(1)) dtdefs.beltLen]);
% % % % pos = mod(tmp',dtdefs.beltLen)';
% % % %     
% % % % % figure
% % % % % ah2 = gca;
% % % % % plot(ppl_tmp,'o')
% % % % % hold on
% % % % % if ~isempty(outliers)
% % % % %     plot(ah2,outliers,ppl_tmp(outliers),'rx')
% % % % % end
% % % % 
% % % % islap = zeros(size(ppl_tmp));
% % % % for iL = 1:numel(ppl_tmp)
% % % %     if ppl_tmp(iL) >= avgppl*(1-dtdefs.thr.encoder) & ppl_tmp(iL) <= avgppl*(1+dtdefs.thr.encoder)
% % % %         islap(iL) = 1;        
% % % %     end
% % % % end
% % % % 
% % % % % plot(ah2,find(islap),ppl_tmp(find(islap))+200,'mv')
% % % % 
% % % % tmp_dist = zeros(size(rota));
% % % % tmp_dist(idxLapOn(1):end) = cumsum(rota(idxLapOn(1):end).*dtdefs.wheel.dpp)+dtdefs.wheel.dpp;
% % % % pos_tmp = mod(tmp_dist',dtdefs.beltLen);
% % % % 
% % % % % plot(ah1,tt,pos_tmp./max(pos_tmp)*max(dtLap),'-','Color',[1 1 1]*.75)
% % % % 
% % % % idxLapOn = idxLapOn(find(islap));
% % % % idxLapOff = idxLapOff(find(islap));
% % % % ppl = [];
% % % % dpp_lap = [];
% % % % for iL = 1:numel(idxLapOn)
% % % % % %     ppl(iL,1) = diff([idxLapOn(iL) idxLapOff(iL)]);
% % % %     ppl(iL,1) = sum(rota(idxLapOn(iL):idxLapOff(iL)));
% % % %     dpp_lap(iL,1) = dtdefs.beltLen/ppl(iL);
% % % % end

%% pos calc etc.
dist_raw = cumsum(rota.*dtdefs.wheel.dpp)+dtdefs.wheel.dpp;
tmp = dist_raw+diff([dist_raw(idxLapOn(1)) dtdefs.beltLen]);
pos = mod(tmp',dtdefs.beltLen)';

% vec for below calc
% % dppVec = ones(size(dist_raw)).*dtdefs.wheel.dpp;
dppVec = zeros(size(dist_raw));
for ii = 1:numel(idxLapOn)-1
% %     here = idxLapOn(ii):idxLapOn(ii+1)-1;
    here = idxLapOn(ii):idxLapOff(ii);
    dppVec(here) = dpp_lap(ii);
end

% correct for actual lap
dist_corr = cumsum(rota.*dppVec)+dtdefs.wheel.dpp;
tmp = dist_corr+diff([dist_corr(idxLapOn(1)) dtdefs.beltLen]); % corrects for incomplete first lap
pos_corr = mod(tmp',dtdefs.beltLen)';
    
    tmpNoPulses = [];
    tmpDpp = [];
    dppVecNew = zeros(size(rota));
    for ii = 1:numel(idxLapOn)-1
        tmpNoPulses(ii,1) = sum(rota(idxLapOn(ii):idxLapOn(ii+1)-1));
        tmpDpp(ii,1) = dtdefs.beltLen/tmpNoPulses(ii);
        dppVecNew((idxLapOn(ii):idxLapOn(ii+1)-1)) = tmpDpp(ii);
    end
    dist_corrNew = cumsum(rota.*dppVecNew)+dtdefs.wheel.dpp;
    tmp = dist_corrNew+diff([dist_corrNew(idxLapOn(1)) dtdefs.beltLen]); % corrects for incomplete first lap
    pos_corrNew = mod(tmp',dtdefs.beltLen)';
    
    dppVecNew(dppVecNew > 0) = dtdefs.wheel.dpp;
    dist_corrNew = cumsum(rota.*dppVecNew)+dtdefs.wheel.dpp;
    tmp = dist_corrNew+diff([dist_corrNew(idxLapOn(1)) dtdefs.beltLen]); % corrects for incomplete first lap
    pos_corrNew = mod(tmp',dtdefs.beltLen)';
    
    cmp = parula(numel(idxLapOn));
%     figure;
%     ah = gca; hold on
    for ii = 1:numel(idxLapOn)-1
        here = idxLapOn(ii):idxLapOn(ii+1);
%         plot(ah,[0:numel(here)-1]./numel(here),pos_corrNew(here),'.-','Color',cmp(ii,:))
    end

  
    
%     figure;
%     subplot(3,1,1)
%     ah = gca;
%     hold on
%     plot(ah,tt,dist_raw,'k-')
%     plot(ah,tt,dist_corr,'r-')
%     subplot(3,1,2)
%     ah = gca;
%     hold on
%     plot(ah,tt([-1e5:1e5]+8400001),pos([-1e5:1e5]+8400001),'k-')
%     plot(ah,tt([-1e5:1e5]+8400001),pos_corr([-1e5:1e5]+8400001),'r-')
%     plot(ah,tt(idxLapHi),0,'db')
%     plot(ah,tt(idxLapOn),5,'sb')
%     linkaxes(gaa,'x')
%     for ii = 1:100:length(pos)-100
%         xlim([0 99]+ii)
%         drawnow
%         pause
%     end

%     figure;
%     subplot(3,1,1)
%     plot(tt,dist_raw,'k')
%     hold on
%     plot(tt,dist_corr,'m')
%     plot(tt(repmat(idxLapOn,1,2)'),repmat([0 max(dist_raw)]',1,numel(idxLapOn)),'-','Color',[.75 .75 .75])
%     subplot(3,1,2)
%     plot(tt,pos,'k')
%     hold on
%     plot(tt,pos_corr,'m')
%     plot(tt(repmat(idxLapOn,1,2)'),repmat([-5 155]',1,numel(idxLapOn)),'-','Color',[.75 .75 .75])
%     linkaxes(gaa,'x')
%     xlim([0 30])

tic
spd = smooth(diff([dtdefs.wheel.dpp;dist_corr]),dtdefs.sf).*dtdefs.sf;
% spd = difftt([dtdefs.wheel.dpp;dist_raw]).*dtdefs.sf;
% spd = diff([dtdefs.wheel.dpp;dist_corr]).*dtdefs.sf;
spd(spd < 0) = 0;
toc

ttis = (floor(tt(1)):ceil(tt(end)))'; % time in sec
distis = interp1(tt,dist_corr',ttis')'; % distance in sec
% tmp = distis+diff([floor(dist_corr(idxLapLo(1)));dtdefs.beltLen]);
% posis =  floor(mod(tmp,dtdefs.beltLen)); % not very good measure
spdis = diff([0;distis]); % speed in sec [cm/s]
spdis(spdis < 0) = 0;

%% speed analyses
% spddefs = speedDefs;
speed = runEval(spd,dtdefs.sf)

%% lap and time analyses PER MIN
mins = lapEval(idxLapOn,dist_corr,spd,tt,dtdefs.sf) 

%% look for galvo / frame sync signal
try 
    galvoCh = data(:,4); % galvo
    galvoCh = lowpass(galvoCh,dtdefs.misc.lowpass);
    galvoCh(galvoCh < dtdefs.thr.galvo) = 0;
    galvoCh(galvoCh >= dtdefs.thr.galvo) = 1;
    tmp = diff([galvoCh]);
    idxFrSyncLo = find(tmp == -1)+1;
    idxFrSyncHi = find(tmp == 1); % frame onset
    paf(:,1) = pos_corr(idxFrSyncHi); % position at frame
%     raf(:,1) = speed.indices.frames.running(idxFrSyncHi); % run at frame
    raf = zeros(size(idxFrSyncHi));
    for iFr = 1:numel(idxFrSyncHi)-1
        here = idxFrSyncHi(iFr):idxFrSyncHi(iFr+1)-1;
        if any(speed.indices.run_cleaned(here) > 0)
            raf(iFr,1) = 1;
        else
            % do nothing
        end
    end    
    laf(:,1) = lapIdx(idxFrSyncHi); % lap at frame
    lonaf = [];
    for iFr = 1:numel(idxFrSyncHi)-1
        here = find(idxLapOn >= idxFrSyncHi(iFr) & idxLapOn < idxFrSyncHi(iFr+1));
        if ~isempty(here)
            lonaf = cat(1,lonaf,[iFr here]);
        end
    end
    loffaf = [];
    for iFr = 1:numel(idxFrSyncHi)-1
        here = find(idxLapOff >= idxFrSyncHi(iFr) & idxLapOff < idxFrSyncHi(iFr+1));
        if ~isempty(here)
            loffaf = cat(1,loffaf,[iFr-1 here]);
        end
    end
    ttaf(:,1) = [0:numel(idxFrSyncHi)-1].*mean(diff(idxFrSyncHi)./dtdefs.sf)+tt(idxFrSyncHi(1));
catch
    disp('No frame data in dataset')
    idxFrSyncHi = [];
    paf = [];
    raf = [];
    laf = [];
    lonaf = [];
    loffaf = [];
    ttaf = [];
end

%% outdata
outdata.info = metadata;

outdata.defaults.treadmill = dtdefs;
outdata.defaults.speed = speed.defs;

outdata.laps = numel(idxLapOn);
outdata.meters = floor(max(dist_corr)/100);
outdata.duration = ceil(range(tt)/60);
outdata.distance_raw = dist_raw;
outdata.distance_corr = dist_corr;
outdata.speed_raw = spd;
outdata.speed_corr = spd; % same
outdata.position_raw = pos;
outdata.position_corr = pos_corr;
outdata.position_atframe = paf;
outdata.time = tt;
outdata.time_atframe = ttaf;
outdata.distance_ds = distis;
outdata.speed_ds = spdis;
outdata.time_ds = ttis;

outdata.speed = deal(speed);
outdata.mins = deal(mins);

outdata.indices = deal(speed.indices);
outdata.indices.lapOnset = idxLapOn;
outdata.indices.lapOffset = idxLapOff;
outdata.indices.lap = lapIdx;
outdata.indices.frames.sync = idxFrSyncHi; 
outdata.indices.run_atframe = raf;
outdata.indices.lap_atframe = laf;
outdata.indices.lapOnset_atframe = lonaf;
outdata.indices.lapOnset_atframe = loffaf;

outdata.speed = rmfield(outdata.speed,{'defs' 'indices'});

outdata.info.when = datestr(now);

%%
trdEval = outdata;
save(outfile,'trdEval');

%% figure
% fh = figure;
% distance
% laps
% speed
% nlaps
% ratio
% meters
% dpm
% tspm
% histo, nbouts, npauses, avg, pk

%% save
% if saveflag
%     save(outfile,outdata)
%     disp('Data saved')
% else 
%     disp('DAQTRAINEVAL data not saved')
% end

disp('Done.')