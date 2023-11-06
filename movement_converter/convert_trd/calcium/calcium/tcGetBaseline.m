function [b,tmp] = tcGetBaseline(x,ds)
%B = TCGETBASELINE(X)
% 
%

if nargin < 2
    ds = 16;
end;

% fprintf('Calculating baseline ');

[nt,ncells]=size(x);

m = mean(x);
xm = bsxfun(@minus,x,m);

tmp.xm = xm;

% cumulative sum to get F trajectory
xint = cumsum(xm);

tmp.xint = xint;

xint2 = tcDecimate(xint,ds);
xint3 = tcDecimate(xint,ds);
% xint3 = xint2;

tmp.xint2 = xint2;
tmp.xint3 = xint3;

N = 4;                 % Order of polynomial fit
F = 21;                % Window length
[x0,x1,x2]=sgolaydiff(xint3,N,F);

tmp.x0 = x0;
tmp.x1 = x1;
tmp.x2 = x2;

b=[];
for icell = 1:ncells
%     fprintf('.');
    % find breakpoints
    bp = crossing(x1(:,icell));
    
    tmp.bp{icell,1} = bp;
    
    % fit piecewise linear model
    xlin = x0(:,icell) - detrend(x0(:,icell),'linear',bp);
    xlinp = diff(xlin)/ds;
    
    tmp.xlin(:,icell) = xlin;
    tmp.xlinp(:,icell) = xlinp;
    
%     plot(x0);hold on;
%     plot(bp,x0(bp),'r.')

    ind = find(xlinp<0);
    
    %b(icell)=median(xlinp(ind));
    b(icell)=prctile(xlinp(ind),10);
end
% fprintf(' Done!\n');

b = b + m;

return;


%%
icell = 1;
x = (cumsum(tcremovedc(tcs2.correct(:,icell))));
k = gausswin(256);k = k/sum(k);
y = (filter(k,1,x));
yp = diff(y);
plot(yp);plot(xlinp(:,1)) 
bp = crossing(yp);
a = y - detrend(y,'linear',bp);
plot(diff(a))
% plot(y);
% hold on;
% plot(a,'r');

b = (diff(a).*(diff(a)<0));
b(find(~b))=nan;
c = nanmedian(b)