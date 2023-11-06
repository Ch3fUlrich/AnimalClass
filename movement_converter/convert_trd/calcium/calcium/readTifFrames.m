
%% 
infile = 'z:\Steffen\Test\compression_3_3.tiff'
outfile = 'z:\Steffen\Test\compression_3_3_seg.tiff'

%%
stack  = Tiff(infile);

[nRows,nCols] = size(stack.read());

% tic
% nFrames = length(imfinfo(infile));
% toc

nFr = nFrames;
nFr = 600;

data = zeros(nRows,nCols,nFr);
data(:,:,1)  = stack.read();

for iFr = 2:nFr
    disp(num2str(iFr))
    stack.nextDirectory()
    data(:,:,iFr) = stack.read();
end

tic
% write2tiff(data,outfile)
writetiff(data,outfile)
toc

