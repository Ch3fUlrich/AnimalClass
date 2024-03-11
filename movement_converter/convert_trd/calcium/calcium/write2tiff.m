function b = write2tiff(array,filename,typestr)
% writes 3D array as tiff file using imageJ code
%	b = write2tiff(array,filename,typestr)
%   
%   200421 SK, 080718 VB

tic;
if nargin < 3
    typestr = class(array);
end

if ~strcmp(class(array),typestr)
    array = cast(array,typestr);
end

[h,w,nframes,nPlanes] = size(array);

fprintf(1,'creating imageplus object\n');

imp = ijarray2plus(array,typestr);

fprintf(1,'writing file\n');

pathstr = fileparts(filename);

if length(pathstr) & exist(pathstr) ~= 7
    mkdir(pathstr);
end
    
b = ijwritetiff(imp,filename);

if ~b
    error('write fail: does directory exist?');
end

t = toc;
s = whos('array');
fprintf('wrote %i frames in %2.1f seconds (%2.0f fps or %5.0f MB/s )\n',nframes,t,nframes/t,s.bytes/1024^2/t);

clear imp
