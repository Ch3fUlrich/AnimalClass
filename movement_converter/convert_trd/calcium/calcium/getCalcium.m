function [timecourses,stat,ops] = getCalcium(s2pfile,fps,overwriteFlag)
% GENERATES (OR LOADS, IF EXSIST) RAW-ISH CALCIUM TRANSIENTS FROM SUIT2P OUTPUT
%   [timecourses,stat,ops] = getCalcium(s2pfile,fps,overwriteFlag)
%   
%   SK 211001 V3, some edits
%   SK 210830 V2


if nargin < 3
    overwriteFlag = 0;
end
if nargin < 2
    fps = [];
end

timecourses = [];
stat = [];
ops = [];

%% defs
% fps = sf;
[fp,n,e] = fileparts(s2pfile);

if (exist(strcat(fp,'\F.mat'),'file') && ~overwriteFlag) || isempty(fps)
    try
        load(strcat(fp,'\fps.mat'));
    catch
        % implement load from mesc here64
        fps = 30.966814176218747; 
        disp('fps set to femtonics 512x512 default (~31 fps)')
    end
else
    % implement load from mesc here64
    save(strcat(fp,'\fps.mat'),'fps');
end

%% F
if exist(strcat(fp,'\F.mat'),'file') && ~overwriteFlag
    load(strcat(fp,'\F.mat'));
else
    load(s2pfile) % suite2p output mat file
    mesc_F = F';
    
    F = double(mesc_F);
    nCells = size(F,2);
    
%     idx = find(sum(F,1) == 0);
%     while ~isempty(idx)
%         for ii = 1:numel(idx)
%             F(:,idx(ii)) = F(:,randi(nCells,1)); % find out why few are zero
%         end
%         idx = find(sum(F,1) == 0);
%     end

    save(strcat(fp,'\F.mat'),'F');
end

idx = find(sum(F,1) == 0);
zeroedROIs = idx;
if ~isempty(idx)
    cc = 1;
    while numel(find(sum(F,1) == 0)) > 0
        try
            F(:,idx) = F(:,idx+1); % find out why few are zero <- mainly because of overlapping ROIs
        catch
            F(:,idx) = F(:,idx-1); % if last ROI
        end
        idx = find(sum(F,1) == 0);
        cc = cc+1;
    end

    save(strcat(fp,'\F.mat'),'F');
end

%% time axis
if exist(strcat(fp,'\tt.mat'),'file') && ~overwriteFlag
    load(strcat(fp,'\tt.mat'));
else    
    tt = ([0:size(F,1)-1]./fps)';
    save(strcat(fp,'\tt.mat'),'tt');
end

%% baselines (adpativebaselines)
if exist(strcat(fp,'\baselines.mat'),'file') && ~overwriteFlag
    load(strcat(fp,'\baselines.mat'));
else
    baselines = zeros(size(F));
    tic;
    for iC = 1:nCells
        disp(['cell ',num2str(iC),' of ',num2str(nCells)])
        try
            baselines(:,iC) = tcGetBaselineAdapt(F(:,iC));
            disp('baseline adapt')
        catch
            % fast dirty
            p = prctile(F(:,iC),0.05);
            baselines(:,iC) = repmat(p,size(F,1));
            disp('prctile baseline')
        end
    end
    toc
    
    % try
    %     baselines = tcGetBaselineAdapt(F);
    % catch
    %     baselines = tcGetBaseline(F);
    % end
    save(strcat(fp,'\baselines.mat'),'baselines');
end

%% dF
if exist(strcat(fp,'\dF.mat'),'file') && ~overwriteFlag
    load(strcat(fp,'\dF.mat'));
else
    try
        dF = (F-baselines)./baselines*100;
    catch
        dF = (F-repmat(baselines,size(F,1),1))./repmat(baselines,size(F,1),1)*100;
    end
    
    % dF_filtered = lowpass(dF,.05);
    
    % tt = [0:size(dF,1)-1]./sf;
    
    save(strcat(fp,'\dF.mat'),'dF');
    % save(strcat(fp,'\dF_filtered.mat'),'dF_filtered');
end

%% deconvolution
if exist(strcat(fp,'\deconv.mat'),'file') && ~overwriteFlag
    load(strcat(fp,'\deconv.mat'));
    deconv = filtfilt(1/round(sf)*ones(1,round(sf)),1,double(spks')); % do it anyway
    
    save(strcat(fp,'\deconv.mat'),'deconv');
else
%     tic;
%     deconv = doDeconvolution(dF,.2,'jovo',1,1/fps);
%     toc
%     decs_tmp1 = doDeconvolution(dF,.05,'jovo',1,1/sf);
%     decs_tmp2 = doDeconvolution(dF,1,'jovo',1,1/sf);
    % deconv_filtered = filtfilt(1/sf*ones(1,round(sf)),1,decs);
    if ~exist('spks','var')
        load(infile,'spks')
    end
    deconv = filtfilt(1/round(sf)*ones(1,round(sf)),1,double(spks'));
    
    save(strcat(fp,'\deconv.mat'),'deconv');
    % save(strcat(fp,'\deconv_filtered.mat'),'deconv_filtered');
end

decs = deconv; % fix

%% Best
% return
timecourses.F = F;
timecourses.baselines = baselines;
timecourses.dF = dF;
% % timecourses.dF_filtered = dF_filtered;
timecourses.deconv = deconv;
% % timecourses.deconv_filtered = deconv_filtered;
timecourses.tt = tt;
timecourses.fps = fps;
timecourses.zeroedROIs = zeroedROIs;
