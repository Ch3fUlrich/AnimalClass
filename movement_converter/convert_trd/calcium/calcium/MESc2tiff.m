function outdata = MESc2tiff(infile)
% outdata = MESc2tiff(infile)
% function outdata = MESc2tiff(expt,nFrPerSeg,invertFlag,flipFlag,parWorkers) % TBD
%   converts femtonics MESc file to tiff
%   reads in green (UG) and red (UR) PMT channel data
%    
%   <infile> is the full path to the measurement
%   <expt> experiment passed as e.g. 'DON-123456_YYYYMMDD_INSTRUMENT_SESSION'  % TBD
%   
%   <outdata> some counters... 
%   
%   See also MESC2TIFFCONVERT, MESC2TIFFDEFS
%   
%   210527 SK V6, minor edits
%   210430 SK V5, fix for multi-channel acq
%   210322 SK V4, some edits, dual channels, idx handling
%   210104 SK V3, bugfix flipping
%   200422 SK V2, added dirs and defs i/o
%   200416 SK V1

% tif output will be segments of format [ LINES x PIXES x CHANNELS x FRAMES ] x N_SEGMENTS

if nargin < 1
%     expt = input('Specify experiment: ','s'); % pass as e.g. 'DON-123456_YYYYMMDD_INSTRUMENT_SESSION' % TBD
    infile = input('Specify full path: ','s');
end

%% fix I/O
% dirs = getDirs(expt); % TBD

outdata = [];
outfile = strcat(infile(1:end-4),'tif')

[fp,n,e] = fileparts(outfile);
if ~exist(fp,'dir')
    mkdir(fp)
    disp('output dir created')
end

%% defs, fix
try
    m2tdefs = MESc2tiffDefs; % loads def params
    
    nFrPerSeg = m2tdefs.nFrPerSeg;
    nFrDec = m2tdefs.nFrDec;
    invertFlag = m2tdefs.invertFlag;
    flipFlag = m2tdefs.flipFlag;
    msessionIdx = m2tdefs.msessionIdx;
    munitIdx = m2tdefs.munitIdx;
    channelIdx = m2tdefs.channelIdx;
    fctChannelIdx = m2tdefs.fctChannelIdx;
    frameIdx = m2tdefs.frameIdx;
    tiffformat = m2tdefs.tiffformat;
    disp('default conversion params loaded')
catch
    nFrPerSeg = 500; % was 512
    nFrDec = 30; % decimate, not used (?)
    invertFlag = 1;
    flipFlag = 1;
    msessionIdx  = 0; % assuming 1 session only
    munitIdx = [];
    channelIdx = [];
    fctChannelIdx = 0;
    frameIdx = 0; % first frame
    tiffformat = 'uint16';
    disp('warning: default params overwritten by backup from inside this function')
end

%% start parallel pool
try
    pool = gcp;
    if pool.NumWorkers ~= m2tdefs.parWorkers
        delete(gcp('nocreate'))
        pool = parpool(m2tdefs.parWorkers); % start parallel worker pool
    else
        disp('parpool already running')
    end
    noParpool = 0; 
catch
    noParpool = 1;
    disp('no parpool running')
%     error('error: something wrong with matlab''s parallel toolbox') % probably not installed
end

%% go to file - TBD
% infile = fullfile(dirs.experiment.imaging.pp.femtonics,files(1).name); % assumes only one mesc file in dir for now
% outfile = fullfile(dirs.experiment.imaging.pp.femtonics_tif,strcat(files(1).name(1:end-4),'tif'));
% 
% try
%     cd(dirs.experiment.imaging.pp.femtonics)
% catch
%     error('error: dir does not exist')
% end
% files = dir('*.mesc');
% if isempty(files)
%     error('error: no mesc files found under this expt ID')
% end
% 
% nFiles = size(files,1);
% if nFiles > 1
%     for iF = 1:nFiles
%         % do sth
%     end
% end
    
%% file info etc.
[fp,n,e] = fileparts(infile);
filename = strcat(n,e);
cd(fp)

i = h5info(infile); % read mesc header
% attribValue = listMEScH5ObjAttribs(filename,'/');

%% mesc to tiff conversion
% data = readMEScMovieFrame(path,msessionIdx,munitIdx,channelIdx,frameIdx);
% % MESc2tiffConvert(infile,nFrPerSeg,invertFlag,flipFlag,parWorkers) % outsource to indep. fct. TBD ?

nUnit = numel(i.Groups.Groups)
if nUnit > 9
    error('too many MUnits - ask SK to fix this :P')
end

unitIDs = [];
for iU = 1:nUnit
    unitIDs(iU) = str2num(i.Groups.Groups(iU).Name(end));
end

if isempty(munitIdx)
    % do nothhing
else
    unitIDs = munitIdx;
    disp('more MUnits in data than selected for conversion')
end

if isempty(channelIdx)
    nCh = size(i.Groups.Groups(1).Datasets,1);
    disp('using all channels available in data')
else
    nCh = 1;
    disp('using first channel available in data')
end

%%   
% fct channel is UG
if nCh > 2
    chStrg = '_MChannel0-1-2';
elseif nCh > 1
    chStrg = '_MChannel0-1';
else
    chStrg = '_MChannel0';
end

this = 1:numel(unitIDs)
% this = 1

clear outdata
outdata = repmat(struct('MUnit',[],'N_tif_segs',[],'lost_segs',[]),numel(unitIDs),1);

tic;

for iU = this
    % waitbar here TBD
    munitIdx = unitIDs(iU);
    dataSize = i.Groups.Groups(iU).Datasets(fctChannelIdx+1).Dataspace.Size;
    fovSize = [dataSize(1) dataSize(2)];
    nFr = dataSize(3); % 1 too many ???
    
    % generate seg idx
    nSeg = ceil(nFr/nFrPerSeg)
    segIdx = [1:nFrPerSeg:nFr nFr+1]-1;
    for iCh = 1:nCh
        frIdx(:,iCh) = iCh:nCh:nFrPerSeg*nCh;
    end
    
%     tmp = nan(fovSize(1),fovSize(2),nCh,nFrPerSeg);
    tmp = nan(fovSize(1),fovSize(2),nFrPerSeg*nCh); % interleaved channels in suite2p
    
    outdata(iU,1).MUnit = munitIdx;
    outdata(iU,1).N_tif_segs = nSeg;
    outdata(iU,1).lost_segs = [];
    
    % % lostFr = zeros(nFrPerSeg,nSeg); % change size/dim. if more than on ch.
    
    cc = nan(nSeg,1);
    % run over frames/segs
    tic
    parfor iSeg = 1:nSeg
        % % tmpFr = zeros(nFrPerSeg,1);
        disp(num2str(iSeg))
        outfile = strcat(infile(1:length(infile)-5),'_MUnit',num2str(munitIdx),chStrg,'_Seg',num2str(iSeg-1),'.tif');
        [fp,n,e] = fileparts(outfile);
        outfile = strcat(fp,'\tif\',n,e);
        data = tmp;
        this = segIdx(iSeg):segIdx(iSeg+1)-1;
        for iFr = 1:numel(this)
            try
                for iCh = 1:nCh
%                     data(:,:,iCh,iFr) = readMEScMovieFrame(filename,msessionIdx,munitIdx,iCh-1,this(iFr));
                    data(:,:,frIdx(iFr,iCh)) = readMEScMovieFrame(filename,msessionIdx,munitIdx,iCh-1,this(iFr));
                end
            catch
                disp('invalid frame idx ?')
                % tmpFr(iSegFr,1) = 1;
            end
        end
        if iFr < nFrPerSeg
%             data = data(:,:,:,1:iFr);
            data = data(:,:,1:iFr*nCh);
        end
        % invert LUT
        if invertFlag
            data = -(data-(2^16-1)); % this may not be needed for old data sets (not really checked)
        end
        % flip left/right
        if flipFlag
            % data = flip(data,2);
            data = flip(data,1); % set to this one 210104 SK
        end
        data = squeeze(data);
        write2tiff(data,outfile,tiffformat) % should work for multi-channel acq. 
        if ~exist(outfile,'file')
            cc(iSeg,1) = iSeg;
            disp('segment lost ?')
        else
            cc(iSeg,1) = 0;
        end
    end
    try
        outdata(iU,1).lost_segs = sort(cc(cc > 0));
    catch
        outdata(iU,1).lost_segs = [];
    end
    if ~isempty(outdata(iU,1).lost_segs)
        warning('SOME SEGS ARE MISSING.')
        pause(5)
    end
    toc
end

toc
disp(infile)
disp('Done.')

