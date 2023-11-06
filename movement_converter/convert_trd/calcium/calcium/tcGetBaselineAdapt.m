function baseline = tcGetBaselineAdapt(timeCourses,blkSiz)
% baseline = tcGetBaselineAdapt(timeCourses)

% 140116 SK - size bug (?) b vs blkInd fixed

[nSamples,nCells]=size(timeCourses);
if nargin <2   % 160825, 4096 might be too large for multiplane imaging
    blkSiz = 4096;
end

stepSiz = 1024;

%%
% nBlks = nSamples/blkSiz;
nBlks = floor(nSamples/blkSiz); % SK, floor added
nSteps = floor((nSamples-blkSiz)/stepSiz)+1;

b=[];blkInd=[];
parfor iStep = 1:nSteps
    sel = stepSiz*(iStep-1) + [1:blkSiz];
    blkInd(iStep)=min(sel);
    b(iStep,:) = tcGetBaseline(timeCourses(sel,:));
end

% blkInd = blkSiz/2:blkSiz:nSamples;
% if length(blkInd) > nBlks % SK added
%     blkInd = blkInd(1:nBlks);
% end
% blkTimes = tcs.tt(blkInd)

f = @(p,tt) p(1)+p(2)*exp(-tt/p(3));

% clf
% plot(blkTimes,b(:,iC)');
% hold on;
%     plot(blkTimes,f(phat,blkTimes));

baseline= zeros(nSamples,nCells);

opt = optimset('Display','off');

%%
parfor iC = 1:nCells
    
%     p1 = polyfit(blkInd,b(:,iC)',1);
%     mse1 = mean((b(:,iC)'-polyval(p1,blkInd)).^2)
    
    startPars = [10,5,500];
    [p2,res] = lsqcurvefit(f,startPars,blkInd,b(:,iC)',[],[],opt);
    mse2 = mean((b(:,iC)'-f(p2,blkInd)).^2);
    f(p2,blkInd);
    
    baseline(:,iC) = f(p2,1:nSamples);

%     clf;
%     plot(1:nSamples,timeCourses(:,iC));
%     hold on;
%     plot(1:nSamples,baseline(:,iC),'r','linewidth',2);
end;

%%
return;


%% 
figure;
plot(baseline(:))