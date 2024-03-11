function m2tdefs = MESc2tiffDefs
% MESc2tiffDefs loads default params for MESC2TIFF
%   
%   210323 SK, edits
%   200422 SK

%% defs
m2tdefs.nFrPerSeg = 500; % n frames per segment
disp('match nFrPerSeg to suite2p param')
m2tdefs.nFrDec = 20; % n frames to decimate
m2tdefs.invertFlag = 1; % invert MESc LUT
m2tdefs.flipFlag = 1; % horz flip of image
m2tdefs.parWorkers = feature('numcores'); % parallel workers = CPU cores
m2tdefs.parWorkers = 8; % test
if m2tdefs.parWorkers > 12
    m2tdefs.parWorkers = 12;
    disp('no. of parallel workers reduced to 12')
end

m2tdefs.msessionIdx  = 0; % assuming 1 session only, 1 session = 1 day
m2tdefs.munitIdx = []; % intentionally empty, assuming all units (=recs) are to be used, 1 unit = 1 rec
m2tdefs.channelIdx = []; % intentionally empty, assuming all channels are to be used
% m2tdefs.channelIdx = 0; % UG, green PMT
% m2tdefs.channelIdx = 1; % UR, red PMT
m2tdefs.fctChannelIdx = 0; % UG, green PMT; functional channel (UR with other label)
m2tdefs.frameIdx = 0; % first frame

m2tdefs.tiffformat = 'uint16'; % standard tif format, other formats may not work with suite2p

% i = gcp;
% if i.NumWorkers ~= m2tdefs.parWorkers
%     pool = parpool(m2tdefs.parWorkers); % start parallel worker pool
% end

