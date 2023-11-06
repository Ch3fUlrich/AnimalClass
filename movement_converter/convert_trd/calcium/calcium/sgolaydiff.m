function [x0,x1,x2]=sgolaydiff(x,N,F)
%SGOLAYDIFF Savitzky-Golay interpolation and differentiation
% [X0,X1,X2]=SGOLAYDIFF(X,N,F) Smooths signal X using a Savitzky-Golay
% filter, X0, and calculates the 1st and 2nd order derivatives, X1 and X2.
%
% The Savitzky-Golay smoothing and differentiation filter optimally fits a
% set of data points to a polynomial in the least-squares sense.


[nsamples,ncells]=size(x);

[b,g] = sgolay(N,F);   % Calculate S-G coefficients

% compute the steady state output, 
x0 = filter(g(:,1),1,x);  % Zero-th derivative (smoothing only)
x1 = -filter(g(:,2),1,x);  % 1st differential
x2 = 2*filter(g(:,3),1,x);  % 2nd differential !!!! NOT CHECKED!!!!

% compensates for delay, the last ((F+1)/2-1) will be gargage.
x0 = circshift(x0,-((F+1)/2-1));
x1 = circshift(x1,-((F+1)/2-1));
x2 = circshift(x2,-((F+1)/2-1));

return;

% similar (but much slower) algorithm
% for icell = 1:ncells
%     
%     for n = (F+1)/2:nsamples-(F+1)/2,
%       % Zero-th derivative (smoothing only)

%       x0(n,icell) =   dot(g(:,1), x(n - HalfWin: n + HalfWin,icell));
% 
%       % 1st differential
%       x1(n,icell) =   dot(g(:,2), x(n - HalfWin: n + HalfWin,icell));
% 
%       % 2nd differential
%       x2(n,icell) = 2*dot(g(:,3)', x(n - HalfWin: n + HalfWin,icell))';
%     end
% 
%     x1 = x1/dx;         % Turn differential into derivative
%     x2 = x2/(dx*dx);    % and into 2nd derivative
% end