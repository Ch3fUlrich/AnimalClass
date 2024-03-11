function ISXD2tiff(infile,outfile,nFrPerSeg)
% READS INSCOPIX MINISCOPE DATAFILE AND SAVES AS TIF
%   ISXD2tiff(infile,outfile,frPerSeg)
%   
%   this requires ISDP to be installed and/or ISDP Matlab packages mapped
%   
%   201203 SK


if nargin < 3
    nFrPerSeg = [];
    nSeg = 1;
end
if nargin < 2
    outfile = [];
end
if isempty(outfile)
    outfile = strcat(infile(1:length(infile)-5),'.tif');
end

%% defs
tiffformat = 'uint16';
% nFrPerSeg = 500;

%% file
vid = isx.Movie.read(infile);
nFr = vid.timing.num_samples;
fovSize = vid.spacing.num_pixels;

if isempty(nFrPerSeg)
    nFrPerSeg = nFr;
end

%% convert
tic
if nSeg > 1
    nSeg = ceil(nFr/nFrPerSeg)
    segIdx = [1:nFrPerSeg:nFr nFr+1]-1;

    tmp = nan(fovSize(1),fovSize(2),nFrPerSeg);
    
%     parfor iSeg = 1:nSeg
    for iSeg = 1:nSeg
        tic
        disp(num2str(iSeg))
        outfile = strcat(infile(1:length(infile)-5),'_Seg',num2str(iSeg-1),'.tif');
        data = tmp;
        this = segIdx(iSeg):segIdx(iSeg+1)-1;
        for iFr = 1:numel(this)
            try
                data(:,:,iFr) = vid.get_frame_data(this(iFr));
% %                 thisframe = vid.get_frame_data(this(iFr));
% %                 imwrite(thisframe,outfile,'writemode','append')
            catch
                disp('sth wrong here')
            end
        end
        write2tiff(data(:,:,1:numel(this)),outfile,tiffformat)
        toc
    end
else
    export_movie_to_tiff(infile,outfile,'write_invalid_frames', true);
end
toc

disp('Done.')


