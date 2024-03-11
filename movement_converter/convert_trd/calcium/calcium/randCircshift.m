function outdata = randCircshift(indata,dim)
% RANDOM CIRCSHIFT 
%   outdata = randCircshift(indata,dim)
%   
%   SK 210825

if nargin < 2
    dim = 1;
end

if dim > 1
    indata = indata';
end

shifts = randi(size(indata,1),1,size(indata,2));
outdata = cell2mat(arrayfun(@(x) circshift(indata(:,x),[shifts(x) 1]),(1:numel(shifts)),'un',0));

if dim > 1
    outdata = outdata';
end
        