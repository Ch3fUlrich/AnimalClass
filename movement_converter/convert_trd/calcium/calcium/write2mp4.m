function write2mp4(inPath,id,cmpRat)
% CREATES MPEG-4 MOVIE FROM TIF FILES IN PATH AND MAYBE DOES COMPRESSION
%   write2mp4(inPath,id,cmpRat)
%   
%   191212 SK

if nargin < 2
    id = [];
end
if nargin < 3
    cmpRat = [];
end

% inPath = 'd:\Steffen\Experiments\DON-000055\20191212\TR-BSL';

if ~strcmp(inPath(end),'\')
    inPath = strcat(inPath,'\');
end
[pathn,~,~] = fileparts(inPath);

D = dir(inPath);
nF = size(D,1);
dates = nan(nF,1);
tiffn = {};
cc = 1;

% tic
for iF = 1:nF
    if contains(D(iF).name,'.tif')
%         stack(:,:,cc) = imread(D(iF).name);
        idx = strfind(D(iF).name,'_');
        tiffn{cc,1} = D(iF).name(1:idx(end)-1);
        cc = cc+1;
        dates(iF) = D(iF).datenum;
    end
end
sessn = unique(tiffn);

[~,si] = sort(dates,'ascend');
si = si(~isnan(dates(si)));
% toc

% tic
if isempty(id)
    here = 1:numel(sessn);
end

for iS = 1:numel(here)
    vob = VideoWriter(strcat(fullfile(pathn,sessn{here(iS)}),'.mp4'),'MPEG-4');
    open(vob);
    idx = find(strcmp(sessn{here(iS)},tiffn));
    for iF = 1:numel(idx)
        frame = imread(fullfile(D(si(idx(iF))).folder,D(si(idx(iF))).name));
        if ~isempty(cmpRat)
            frame = imresize(frame,.5);
        end
        writeVideo(vob,frame);
    end
    close(vob);
end

% toc
 
 