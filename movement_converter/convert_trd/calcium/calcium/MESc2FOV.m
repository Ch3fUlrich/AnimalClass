function [stack,fov,fh] = MESc2FOV(infile,dFr)
% READS N-th FRAME FROM MESc FILE AND GENERATES FOV PREVIEW
%   assumes UG (green ch) has been acquired exclusively
%   [stack,fov,fh] = MEScFOV(infile,dFr)
%
%   SK 210104

if nargin < 2, dFr = 30; end
stack = [];
fov = [];

% infile = 'd:\Steffen\scratch\DON-003484\20201230\002P-F\DON-003484_20201230_002P-F_S1-S2-S3-ACQ.mesc'

%% defs
msessionIdx = 0; % assuming N=1 sessions
channelIdx = 0; % green PMT
invertFlag = 1;
flipFlag = 1;

CLims = [0 5e3];

%%
[fp,n,e] = fileparts(infile);
filename = strcat(n,e);
cd(fp)

i = h5info(infile);
nUnit = numel(i.Groups.Groups)
unitIDs = [];
for iU = 1:nUnit
    unitIDs(iU) = str2num(i.Groups.Groups(iU).Name(end));
end

%%
this = 1:numel(unitIDs)

%%
for iU = this
    clear tmp avg
    munitIdx = unitIDs(iU);
    dataSize = i.Groups.Groups(iU).Datasets.Dataspace.Size;
    fovSize = [dataSize(1) dataSize(2)];
    nFr = dataSize(3) % 1 too many ???
    
    theseFr = 1:dFr:nFr;
    tmp = zeros(fovSize(1),fovSize(2),numel(theseFr));
    
    tic
    for iFr = 1:numel(theseFr)
        disp(strcat('frame ',num2str(iFr)))
        clear fr
        fr = double(readMEScMovieFrame(filename,msessionIdx,munitIdx,channelIdx,theseFr(iFr)));
         if invertFlag
            fr = -(fr-(2^16-1));
        end
        if flipFlag
            fr = flipud(fr);
        end
        tmp(:,:,iFr) = fr;
    end
    toc
    avg = nanmean(tmp,3);
    if isscalar(this)
        stack = tmp;
        fov = avg;
    else
        stack{iU,1} = tmp;
        fov{iU,1} = avg;
    end
    fh(iU) = figure;
    imagesc(avg,minimax(avg(:)));
    axis image
    axis off
    cbh = colorbar;
    ylabel(cbh,'F')
    set(cbh,'TickLength',0)
    colormap(gray(64))
end

return

%%
tmp = [];
for iU = 1:size(stack,1)
    tmp = cat(3,tmp,stack{iU});
end

% stackAnimator(stackGroupProject(tmp,30),'FramesPerSec',5)%,'CLim',[0e3 10e3])

writetiff(tmp,'c:\Users\rg-fd02-user\Desktop\stack2.tif','uint16')
