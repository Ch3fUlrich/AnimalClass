function readMESc(infile,nFrPerSeg,invertFlag,flipFlag,parWorkers)
% converts femtonics MESc file to tiff
%   readMESc(infile)
%
%   200416 SK V1

if nargin < 1
    infile = input('specify full path to MESc file: ','s');
end
if nargin < 2
    nFrPerSeg = 512; % n frames per segment
end
if nargin < 3
    invertFlag = 0;
end
if nargin < 4
    flipFlag = 0;
end
if nargin < 5
    parWorkers = feature('numcores');
end

%% more defs
% parWorkers = 4;
msessionIdx  = 0; % assuming 1 session only
tiffformat = 'uint16';

%% I/O
dirs = getDirs;

% infile = 'd:\Steffen\scratch\ten_sec_test.mesc';
% infile = 'd:\Steffen\scratch\DON-001366\20200324\002P-F\DON-001366_20200324_002P-F_S1-ACQ.mesc'
% infile = 'd:\Steffen\scratch\DON-001368\DON-001368_20200416_002P-F_S1-ACQ.mesc';
% infile = 'd:\Steffen\scratch\DON-001368_20200418_002P-F_S1-S2-ACQ\DON-001368_20200418_002P-F_S1-S2-ACQ.mesc';

outfile = strcat(infile(1:end-4),'tif')

%% parallel workers
i = gcp;
if i.NumWorkers ~= parWorkers
    pool = parpool(parWorkers); % check for open workers
end

%% file info etc.
[fp,n,e] = fileparts(infile);
filename = strcat(n,e);
cd(fp)

i = h5info(infile);
% attribValue = listMEScH5ObjAttribs(filename,'/');

%% mesc to tiff conversion
% data = readMEScMovieFrame(path,msessionIdx,munitIdx,channelIdx,frameIdx);

nUnit = numel(i.Groups.Groups);
for iU = 1:nUnit
    dataSize = i.Groups.Groups(1).Datasets.Dataspace.Size;
    fovSize = [dataSize(1) dataSize(2)];
    nFr = dataSize(3); % 1 too many ???
    % nFr = 10000;
    munitIdx = iU-1; rea
    channelIdx = 0; % UG
    % channelIdx = 1; % UR
    frameIdx = 0; % first frame

    % generate seg idx
    nSeg = ceil(nFr/nFrPerSeg);
    segIdx = [1:nFrPerSeg:nFr nFr+1];

    tmp = nan(fovSize(1),fovSize(2),nFrPerSeg);
    
    % run over frames/segs
    tic
    parfor iSeg = 1:nSeg
        disp(num2str(iSeg))
        outfile = strcat(infile(1:end-5),'_',num2str(iSeg),'.tif');
        data = tmp;
        this = segIdx(iSeg):segIdx(iSeg+1)-1;
        for iFr = 1:numel(this)
            data(:,:,iFr) = readMEScMovieFrame(filename,msessionIdx,munitIdx,channelIdx,this(iFr));
        end
        if iFr < nFrPerSeg
            data = data(:,:,1:iFr);
        end
        if invertFlag
            data = -(data-(2^16-1));
        end
        if flipFlag
            data = flip(data,2);
        end
        writetiff(data,outfile,tiffformat)
    end
    toc
end


