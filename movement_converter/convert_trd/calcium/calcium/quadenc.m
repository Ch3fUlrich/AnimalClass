function out = quadenc(cha,chb)
% QUADENC calulates rotation direction of quadrature rotary encoder 
%   out = quadenc(cha,chb)
%   
%   <cha,chb> two encoder channels with Hi (1) and Lo (0) states
%   <out> rotation vector with fwd (1) and rev (-1) motion, same size as input
%   
%   200528 kandler

cha = cha(:);
chb = chb(:);

% enc-state look-up table
lut = [...
    0 0;... % state 1
    1 0;... % state 2
    1 1;... % state 3
    0 1;... % state 4
    ];  

[~,statevec] = ismember([cha chb],lut,'rows'); % this does sth like FIND

out = zeros(size(statevec));

out(diff(statevec) == 3) = -1; % rev
out(diff(statevec) == -3) = 1; % fwd

return

%% slow version
% statevec = nan(size(cha));
% parfor ii = 1:numel(cha)
%     [~,statevec(ii,1)] = ismember([cha(ii) chb(ii)],lut,'rows');
% end
% 
% out = zeros(size(statevec));
% parfor ii = 2:numel(statevec)
%     d = diff(statevec(ii-1:ii));
%     switch d
%         case 3
%             r = -1;
%         case -3
%             r = 1;
%         otherwise
%             r = 0;
%     end
%     out(ii,1) = r;
% end
