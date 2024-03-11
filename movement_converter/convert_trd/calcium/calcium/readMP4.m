function stack = readMP4(videofile,framesToRead)
% READS FRAMES FROM MP4 VIDEO OBJECT
%   stack = readMP4(videofile,framesToRead)
%   <framesToRead> can be a scalar or vector of indices
%   assumes <videofile> is a B/W video (not RGB)
%   
%   210913 SK V2, some frame idx edits

tic
obj = VideoReader(videofile); % vob
toc

if nargin < 2
    framesToRead = obj.NumberOfFrames;
end

if isscalar(framesToRead)
    frameIdx = 1:framesToRead;   
end

img = read(obj,1);
stack = [];
tic
stack = zeros(obj.Height,obj.Width,numel(frameIdx),class(img)); % output of same class as vob
% stack = zeros(obj.Height,obj.Width,numel(frameIdx),3,class(img)); % RGB
toc

disp(['reading frames ',num2str(frameIdx(1)),' to ',num2str(frameIdx(end))])
% tic
wb = waitbar(0);
h = get(get(wb,'Children'));
for iFr = 1:numel(frameIdx)
    waitbar(iFr/numel(frameIdx),wb);
    if ~mod(iFr,100)
        h.Title.String = [num2str(iFr),'/',num2str(numel(frameIdx))];
    end
    img = read(obj,frameIdx(iFr));
    stack(:,:,iFr) = squeeze(img(:,:,1)); % squeeze B/W video
%     stack(:,:,iFr,:) = img; % RGB
end
% toc
close(wb)
disp('done.')
