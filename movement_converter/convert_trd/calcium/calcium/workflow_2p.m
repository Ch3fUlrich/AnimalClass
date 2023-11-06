
%% experiment 
% expt = 'DON-001234_20210101_002P-F_S1-S2-S3-ACQ';
mescfile = 'd:\Users\Username\scratch\DON-001234\20210101\002P-F\DON-001234_20210101_002P-F_S1-S2-S3-ACQ.mesc';

user = 'Username';


%% generate data paths
% dirs
[filePath,expt,fileExt] = fileparts(mescfile)

dirs = getDirs(expt,user);
% % filenames = getFilenames(expt,dirs);

% file logic
% files
% 2p calcium mesc
% mousecam mp4
% behaviour mat / csv

%% conversion
% mesc to tif
MESc2tiff(mescfile);
% MESc2tiff(expt);
% info = MESc2tiff(mescfile);

%% denoising of reg'ed tiffs ?
% % deep-interpol ==> python / sciCORE ?


%% suite2p
% =====> manually curated <========

% --> python and sciCORE ?
% move dir

% mkdir
% copyfile
% rmdir

% outputs: 
% infile = 'd:\Users\Username\scratch\Steffen\DON-001234\20210101\002P-F\tif\suite2p\plane0\Fall.mat'
% outfile = 'd:\Users\Username\scratch\DON-001234\20210101\spks.npy';
% outfile_snr = 'd:\Users\Username\scratch\DON-001234\20210101\snr.csv';

% s2pfile = 'd:\Users\Username\scratch\DON-001234\20210101\002P-F\tif\suite2p\plane0\Fall.mat';
s2pfile = strcat(filePath,'\tif\suite2p\plane0\Fall.mat')
load(s2pfile)

whos

%% dF & tcs
% % ?
% mescfile = '';

[sf,frTime] = getSF2p(mescfile)
% sf = 31;
sf = sf(1)
fps = sf; 
% sf = 30.966814176218747; % if FOV = 512x512 pxl

overwriteFlag = 0; % redo all calcs on raw F
s2pfile = strcat(filePath,'\tif\suite2p\plane0\Fall.mat');

% load transients 
% [timecourses,stat,ops] = getCalcium(s2pfile,fps,overwriteFlag);
getCalcium(s2pfile,fps,overwriteFlag);
getCalcium;

% timecourses.zeroedROIs;
idx = find(sum(F,1) == 0);
zeroedROIs = idx;
if ~isempty(idx)
    cc = 1;
    while numel(find(sum(F,1) == 0)) > 0
        try
            F(:,idx) = F(:,idx+1); % overlapping ROIs
        catch
            F(:,idx) = F(:,idx-1); % if last ROI
        end
        idx = find(sum(F,1) == 0);
        cc = cc+1;
    end
end

% fig

%% mask
% mask = genS2Pmask(stat,ops);

%% event detection
EV = detectEvents(decs,dF);

ev_thr(:,zeroedROIs) = NaN;
true_fwd(:,zeroedROIs) = NaN;
true_rev(:,zeroedROIs) = NaN;
true_peaks(:,zeroedROIs) = NaN;

% save events

% event fig

%% behaviour
% parse treadmill readout
% laps
% running vs stillness
% stats etc. 
% sub-file handling
trdfiles = {...
    'f:\Users\Username\scratch\DON-001234\20210101\TRD-2P\DON-001234_20210101_TRD-2P_S1-ACQ.mat';...
    'f:\Users\Username\scratch\DON-001234\20210101\TRD-2P\DON-001234_20210101_TRD-2P_S2-ACQ.mat';...
    'f:\Users\Username\scratch\DON-001234\20210101\TRD-2P\DON-001234_20210101_TRD-2P_S3-ACQ.mat';...
    };

overwriteFlag = 1; % 1=redo
misses = [];
for iF = 1:numel(trdfiles)
    thisfile = trdfiles{iF};
    try
        daqTrainEval(thisfile,overwriteFlag);
    catch
        disp('trdfile miss')
        misses = cat(1,misses,iF);
    end
end

%% SNR
% SNR = calcSNR(dF,decs,act_bin,peak_ampl)
% SNRwrapper(expt,decs)

mostActCells = sum(decs,1);
[~,actIdx] = sort(mostActCells,'descend');

resp_win = act_bin;
resp_win(resp_win == 0) = NaN;
noise_win = abs(act_bin-1);
noise_win(noise_win == 0) = NaN;

spk_mean = nanmean(decs,1)'; % not used
spk_sd = nanstd(decs,0,1)'; % not used

% peak_mean = nanmean(pk_ampl);
peak_num = sum(~isnan(peak_ampl),1)';

noise_mean = nanmean(dF.*noise_win,1)';
noise_median = nanmedian(dF.*noise_win,1)';
noise_sd = nanstd(dF.*noise_win,0,1)';
peak_mean = nanmean(dF.*resp_win,1)';
peak_sd = nanstd(dF.*resp_win,0,1)';
peak_mean_from_max = nanmean(peak_ampl)';
dF_mean = nanmean(dF,1)';

cell_snr = peak_mean./noise_sd;
cell_snr_peak = peak_mean_from_max./noise_sd;

%% export

outfile = strcat(fileparts(infile),'\spks.npy');
outfile_snr = strcat(fileparts(infile),'\snr.csv');

peak_thr = 10; % min no of peaks

% writeNPY(decs_filt',outfile);
% writeNPY(dF',outfile);
writeNPY(decs',outfile);

var_names = {'idx' 'session_name' 'cell_id' 'spike_filter_id' 'noise_calc_id' 'snr_df_f' ... % same as in horsto
    'snr_df_f_peak','noise_mean' 'noise_sd' 'dF_mean' 'peak_mean' 'peak_sd' 'peak_mean_from_max'}; % further vars
    
cell_id = [0:nCells-1]'; % criterion
cell_id = find(peak_num >= peak_thr)-1;

idx = [0:numel(cell_id)-1]';
ses_name = repmat(expt,numel(cell_id),1);
spk_filt = repmat('n/a',numel(cell_id),1);
noise_calc = repmat('n/a',numel(cell_id),1);

T = table(...
    idx,ses_name,cell_id,spk_filt,noise_calc,cell_snr(cell_id+1),...
    cell_snr_peak(cell_id+1),noise_mean(cell_id+1),noise_sd(cell_id+1),dF_mean(cell_id+1),peak_mean(cell_id+1),peak_sd(cell_id+1),peak_mean_from_max(cell_id+1),...
    'VariableNames',var_names);

writetable(T,outfile_snr,'Delimiter',',','WriteVariableNames',true,'FileType','text')

%% Ca maps
% projection to trial activity vs position 
% maps = calcMaps()

posVec = [];
posVecSes = [];
runFrames = [];
segIdx = []; % parts
lapOnIdx = []; % lap onsets
lapIdx = []; % laps
corrMat = 0;
posVecSes = [];
lapOnIdxSes = [];
spdMap = [];


% speed map
for iP = 1:numel(trdfiles)
    thisfile = trdfiles{iP};
    outfile = strcat(thisfile(1:end-4),'_eval.mat');
    try 
       load(outfile) 
    catch
        disp('no file')
%         trdEval = daqTrainEval(thisfile);
    end
    cmpb = trdEval.defaults.treadmill.beltLen/100; % N=100 bins changed from 99
    posBins = 0:cmpb:trdEval.defaults.treadmill.beltLen;
    posBins = posBins(1:end-1); % kill last
%     posBins(end) = posBins(end)+trdEval.defaults.treadmill.wheel.dpp; % needed ?

    posVec = cat(1,posVec,trdEval.position_atframe(:));
    posVecSes = cat(1,posVecSes,ones(size(trdEval.position_atframe(:)))*iP);
    runFrames = cat(1,runFrames,trdEval.indices.run_atframe(:));
    segIdx(iP,:) = [1 numel(trdEval.position_atframe)]+corrMat;
    lapOnIdx = cat(1,lapOnIdx,trdEval.indices.lapOnset_atframe(:,1)+corrMat);
    lapOnIdxSes = cat(1,lapOnIdxSes,ones(size(trdEval.indices.lapOnset_atframe(:,1)))*iP);
    lapIdx = cat(1,lapIdx,trdEval.indices.lap_atframe(:));
    corrMat = corrMat+numel(trdEval.position_atframe);
    %         save(outfile,'trdEval');
    
    tmpMap = zeros(numel(posBins),numel(trdEval.indices.lapOnset)-1);
    cc = 0;
% %     tmpIdxMap = zeros(numel(posBins),numel(trdEval.indices.lapOnset)-1);
    for iL = 1:numel(trdEval.indices.lapOnset)-1
        here = trdEval.indices.lapOnset(iL):trdEval.indices.lapOnset(iL+1)-1;
        op = trdEval.position_corr(here);
        [~,pIdx] = histc(op,posBins);
        s = trdEval.speed_raw(here);
        r = trdEval.indices.run_cleaned(here);
%         s = s.*r;
        for iB = 1:numel(posBins)
            try
                idx = find(pIdx == iB);
                if ~isempty(idx)
                    tmpMap(iB,iL) = nanmean(s(idx));
% %                     tmpIdxMap(iB,iL) = nanmean(r(idx));
                else
                    cc = cc+1;
                end
            catch
                % do nothing
%                 keyboard
            end
        end
    end
    tmpMap(isnan(tmpMap)) = 0;
    spdMap = cat(2,spdMap,tmpMap);
% %     tmpIdxMap(isnan(tmpIdxMap)) = 0;
% %     runIdxMap = cat(2,runIdxMap,tmpIdxMap);

end
% need to take the galvo frames ?
sesOnIdx = find(diff([0;lapOnIdxSes]) ~= 0);

% calc Map
calcMap;

%% spatial information
% SI = calcSI()

allSI = [];
si = [];
for iC = 1:nCells
    disp(['cell ',num2str(iC)])
    tic
    this = mapsEvents(:,:,iC);
    this(isnan(this))= 0;
    this = filtfilt(ones(1,3),1,this);
    % this = randCircshift(this,1);
    
    this = mapsDec(:,:,iC);
    this = lowpass(this,.1);
    this(this < 0) = 0;
    
    [nBins,nVars] = size(this);
    
    % this = lowpass(this);
    
    m = [];
    mSpd = [];
    si = [];
    [nBins,nTrials] = size(this);
    here = [sesOnIdx;nTrials+1];
    for iS = 1:numel(sesOnIdx)
        m(:,iS) = nanmean(this(:,here(iS):here(iS+1)-1),2);
        mSpd(:,iS) = nanmean(spdMap(:,here(iS):here(iS+1)-1),2);
    end
    mSpd(end,:) = mSpd(1,:); % dumb fix
    
    Pocc = 1./mSpd;
    Pocc = Pocc./repmat(sum(Pocc,1),[nBins 1]);
    
    
    % activity threshold: min2max (30%), snr, sd
    % width thresh (15-120cm)
    % PC peak thresh (1/3 peaks in-field)
    
    % occup. prob.
    
    % spatial info over bins
    here = [sesOnIdx;nTrials+1];
    for iB = 1:nBins
        for iS = 1:numel(sesOnIdx)
            si(iB,iS) = Pocc(iB,iS)*(m(iB,iS)/nanmean(m(:,iS),1))*log2(m(iB,iS)/nanmean(m(:,iS),1)); % Skaggs
            %         si(iB,iS) = Pocc(iB,iS)*(m(iB,iS))*log2(m(iB,iS)/nanmean(m(:,iS),1));
        end
    end
    % si = nansum(abs(si),1)
    si = nansum(si,1);
    allSI(:,iC) = si;
    toc
end

nShuff = size(mapsShuffDec,4);

allShuffSI = [];
for iC = 1:nCells
    for iShuff = 1:nShuff
        disp(['cell ',num2str(iC)])
        tic
        this = squeeze(mapsShuffEv(:,:,iC,iShuff));
        this(isnan(this))= 0;
        this = filtfilt(ones(1,3),1,this);
        % this = randCircshift(this,1);
        
        this = squeeze(mapsShuffDec(:,:,iC,iShuff));
        this = lowpass(this,.1);
        this(this < 0) = 0;
        
        [nBins,nVars] = size(this);
        
        % this = lowpass(this);
        
        m = [];
        mSpd = [];
        si = [];
        [nBins,nTrials] = size(this);
        here = [sesOnIdx;nTrials+1];
        for iS = 1:numel(sesOnIdx)
            m(:,iS) = nanmean(this(:,here(iS):here(iS+1)-1),2);
            mSpd(:,iS) = nanmean(spdMap(:,here(iS):here(iS+1)-1),2);
        end
        mSpd(end,:) = mSpd(1,:); % dumb fix
        
        Pocc = 1./mSpd;
        Pocc = Pocc./repmat(sum(Pocc,1),[nBins 1]);
        
        
        % activity threshold: min2max (30%), snr, sd
        % width thresh (15-120cm)
        % PC peak thresh (1/3 peaks in-field)
        
        % occup. prob.
        
        % spatial info over bins
        here = [sesOnIdx;nTrials+1];
        for iB = 1:nBins
            for iS = 1:numel(sesOnIdx)
                si(iB,iS) = Pocc(iB,iS)*(m(iB,iS)/nanmean(m(:,iS),1))*log2(m(iB,iS)/nanmean(m(:,iS),1)); % Skaggs
                %         si(iB,iS) = Pocc(iB,iS)*(m(iB,iS))*log2(m(iB,iS)/nanmean(m(:,iS),1));
            end
        end
        % si = nansum(abs(si),1)
        si = nansum(si,1);
        allShuffSI(:,iC,iShuff) = si;
    end
end

% return

%% trial based
% m = [];
% mSpd = [];
% si = [];
% [nBins,nTrials] = size(this);
% allSItrials = [];
% 
% for iC = 1:nCells
%     disp(num2str(iC))
%     this = mapsEvents(:,:,iC);
%     for iTr = 1:nTrials
%         m = this(:,iTr);
%         mSpd = spdMap(:,iTr);
%         mSpd(end) = mSpd(1);
%         Pocc = 1./mSpd;
%         Pocc = Pocc./repmat(sum(Pocc,1),[nBins 1]);
%         for iB = 1:nBins
%             si(iB,iTr) = Pocc(iB)*(m(iB)/nanmean(m,1))*log2(m(iB)/nanmean(m,1));
%         end
%     end
%     si = nansum(si,1);
%     allSItrials(:,iC) = si';
% end

%% place cell metrics
% N
% size
% reliability
% stability 

binw = 1.8;
N = [];
A = [];
W = [];
mA = [];
mW = [];
for iC = 1:nCells
    this = repmat(squeeze(mapsBin(:,:,iC)),2,1);
    that = repmat(squeeze(mapsEvents(:,:,iC)),2,1);
    [~,nTrials] = size(this);
    % this = circshift(this,[40 0]);
    % figure;
    % imagesc(this')
    % hold on
    for iTr = 1:nTrials
        dt = diff(this(:,iTr),1,1);
        a = find(dt == 1)+1;
        if this(1,iTr) == 1
            a = [1;a]; 
        end
        b = find(dt == -1)+1;
        N(iC,iTr) = 0;
        if ~isempty(a)
            for iP = 1:numel(a)/2
                N(iC,iTr) = N(iC,iTr)+1;
                if b(iP) > a(iP)
                    A{iC,iTr}(iP) = max(that(a(iP):b(iP),iTr));
                    W{iC,iTr}(iP) = (b(iP)-a(iP))*binw;
                    %                 try plot(a(iP),iTr,'k>'); end
                    %                 try plot(b(iP),iTr,'k<'); end
                    
                elseif a(iP) > b(iP)
                    A{iC,iTr}(iP) = max(that(a(iP):b(iP+1),iTr));
                    W{iC,iTr}(iP) = (b(iP+1)-a(iP))*binw;
                    %                 try plot(a(iP),iTr,'k>'); end
                    %                 try plot(b(iP+1),iTr,'k<'); end
                else
                end
            end
        else
            A{iC,iTr} = [];
            W{iC,iTr} = [];
        end
        if cell_snr(iC) < 2
            A{iC,iTr} = [];
            W{iC,iTr} = [];
            N(iC,iTr) = 0;
        end
        mA(iC,iTr) = nanmean(A{iC,iTr});
        mW(iC,iTr) = nanmean(W{iC,iTr});
    end
end

% figure;
% ah1 = subplot(3,1,1);
% plot(ah1,N,'ko')
% ah2 = subplot(3,1,2);
% hold on
% ah3 = subplot(3,1,3);
% hold on
% for iTr = 1:nTrials
%     try plot(ah2,iTr,A{iTr}(:),'ko'); end
%     try plot(ah3,iTr,W{iTr}(:),'ko'); end
% end

%%
calcPCmetrics;

PCmetrics(13).width_bins
PCmetrics_1sthalf(13).width_bins
PCmetrics_2ndhalf(13).width_bins
PCmetrics_odd(13).width_bins
PCmetrics_even(13).width_bins

this = PCmetrics;
% these = actIdx(1:200);
[~,actIdx] = sort(nansum(squeeze(nansum(mapsEvents,1)),1),'descend');
these = actIdx(1:200);

binw = 1.8; % [cm]

figure;
ah1 = subplot(3,1,1);
hold on
ah2 = subplot(3,1,2);
hold on
ah3 = subplot(3,1,3);
hold on
cc = 1;
for iC = 1:numel(these)
%     clf;
%     this = mapsEvents(:,:,these(iC));
%     this(isnan(this)) = 0;
%     % this = filter(ones(1,5),1,this);
%     this = filtfilt(ones(1,5),1,this);
%     imagesc(this',[0 3])
%     axis image
%     drawnow
%     pause
% end

    nPk = zeros(1,nSes);
    pkAmpl = [];
    pkWidth = [];
    for iS = 1:nSes
        n = this(these(iC)).N_true{iS};
        if ~isempty(n)
            for iP = 1:numel(n)
                if n(iP) > 0
                    nPk(1,iS) = nPk(iS)+1;
                    pkAmpl = cat(2,pkAmpl,[this(these(iC)).ampl{iS}(iP);iS]);
                    pkWidth = cat(2,pkWidth,[this(these(iC)).width_bins{iS}(iP)*binw;iS]);
                    %                 pkAmpl{iS,1}(iP) = this(these(iC)).ampl{iS}(iP);
                    %                 pkWidth{iS,1}(iP) = this(these(iC)).width_bins{iS}(iP)*binw;
                else
                    pkAmpl = cat(2,pkAmpl,[NaN;iS]);
                    pkWidth = cat(2,pkWidth,[NaN;iS]);
                    %                 pkAmpl{iS,1}(iP) = NaN;
                    %                 pkWidth{iS,1}(iP) = NaN;
                end
            end
        else
            pkAmpl = nan(2,1);
            pkWidth = nan(2,1);
        end
    end
    plot(ah1,[-.1 0 .1]+iC,nPk,'-')
    plot(ah2,pkAmpl(2,:)',pkAmpl(1,:)','o')
    plot(ah3,pkWidth(2,:)',pkWidth(1,:)','o')
    
    %     plot(ah2,1:nSes,nPk,'ko-')
    cc = cc+1;
end
  
%%
allAvg = [];
allTrials = [];
cellIdx = [];
idx = [sesOnIdx;nTrials+1];
% this = maps;
% this = mapsDec;
this = mapsEvents;
% this = mapsBin;
for iS = 1:numel(idx)-1
    cc = 0;
    for iC = 1:numel(these)
        here = idx(iS):idx(iS+1)-1;
        allAvg(:,iC,iS) = squeeze(nanmean(this(:,here,these(iC)),2));
        allTrials{iS,1}(:,[1:numel(here)]+cc) = this(:,here,these(iC));
        cellIdx{iS,1}([1:numel(here)]+cc,1) = iC;
        cc = cc+numel(here);
    end
end
[maxAvg,maxIdx] = max(allAvg,[],1);
maxIdx = squeeze(maxIdx);

for iS = 1:nSes
    [~,sIdx(:,iS)] = sort(maxIdx(:,iS),'ascend');
end

this = allAvg./repmat(maxAvg,100,1,1);

figure;
cc = 1;
for iS = 1:nSes
    for jS = 1:nSes
        subplot(nSes,nSes,cc)
        imagesc(squeeze(this(:,sIdx(:,iS),jS))',[0 1.5])
        axis image
        cc = cc+1;
    end
end
colormap hot

figure;
cc = 1;
mTrials = diff(idx);
for iS = 1:nSes
    for jS = 1:nSes
        here = cellIdx{jS};
        here(:,2) = 1:numel(here);
        %     here2 = repmat(sIdx(:,iS),1,mTrials(iS))';
        %     here2 = here2(:);
        here2 = [];
        for iC = 1:numel(these)
            tmp = find(here(:,1) == sIdx(iC,iS));
            here2 = cat(1,here2,here(tmp,2));
        end
        %         subplot(1,nSes*nSes,cc)
        subax(1,nSes*nSes,cc)
        this = allTrials{jS};
        [maxAvg,~] = max(this,[],1);
        this = this./repmat(maxAvg,100,1);    
        imagesc(squeeze(this(:,here2(1:5:end)))',[.25 1.])
%         imagesc(squeeze(this(:,here2(1:2:end)))',[0 3])
%         axis image
        set(gca,'XTick',[],'YTick',[])
        cc = cc+1;
    end
end
colormap hot    

%% correlations, pairwise, higher-order analyses
% calcMapCorr; 

%% sequence analyses

%% dim reduction and manifols

%% behaviour cont'd
% DLC pupil size, arousal 
% DLC eye movements 
% DLC other behaviours ?





