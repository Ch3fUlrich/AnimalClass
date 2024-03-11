function EV = detectEvents(deconv,dF)
% EVENT DETECTION ON CALCIUM DATA
%   EV = detectEvents(deconv,dF)
%   
%   210709 SK


%% def 
binaThr = 1.5; % binarization of deconv
xsd = 2; % peak detection x-times SD (on dF)

samples_forw_in_sec = 0.2; % post event win
samples_backw_in_sec = 0.1; % pre event win

%% thresholding based on xSD
[nSamples,nCells ] = size(deconv); % use deconvolved

usethis = deconv;

deconv_sd = nanstd(usethis,0,1);
deconv_thr = zeros(size(usethis));
% tmp_thr = zeros(size(usethis));
for iC = 1:nCells
    here = [];
    here = find(usethis(:,iC) > deconv_sd(iC)*binaThr);
    if ~isempty(here)
        deconv_thr(here,iC) = usethis(here,iC);
    end
end

%% binarization
usethis = deconv_thr;

dt = diff(usethis,1,1); 
dt(dt > 0) = 1;
dt(dt < 0) = -1;
tmpl = zeros(round(sf),1);
tmpl(end) = 1;
ev_fwd = nan(size(usethis));
ev_rev = nan(size(usethis));

%% candidate events
for iC = 1:nCells
    disp(num2str(iC))
    % forward dir
    this = dt(:,iC);
    idx1 = strfind(this',tmpl');
    if ~isempty(idx1)
        for ii = 1:numel(idx1)
            try
                ev_fwd(idx1(ii)+numel(tmpl),iC) = 0;
            catch
                % nothing to do here
            end
        end
    end
    % reverse dir
    that = flipud(dt(:,iC));
    idx2 = strfind(that',-1*tmpl');
    if ~isempty(idx2)
        for ii = 1:numel(idx2)
            try
                ev_rev(idx2(ii)+numel(tmpl),iC) = 0;
            catch
                % nothing to do here
            end
        end
    end
end
ev_rev = flipud(ev_rev);        
    
%% refine detection
true_fwd = nan(size(ev_fwd));
true_rev = nan(size(ev_rev));
true_peaks = nan(size(dF));
peak_ampl = nan(size(dF));
act_bin = zeros(size(dF));

dF_sd = nanstd(dF,0,1);

cc = 0;
for iC = 1:nCells
    disp(num2str(iC))
    this = ev_fwd(:,iC);
    that = ev_rev(:,iC);
    tc = lowpass(dF(:,iC),.05);
%     tc = dF(:,iC);
    idx1 = find(this == 0);
    idx2 = find(that == 0);
    if ~isempty(idx1) && ~isempty(idx2)
        for iE = 1:numel(idx1)
            try
                idx3 = find(idx2 > idx1(iE),1,'first');
                if ~isempty(idx3)
                    true_fwd(idx1(iE),iC) = 1;
                    true_rev(idx2(idx3),iC) = 1;
                    here = idx1(iE):idx2(idx3);
                    [pk,ii] = max(tc(here));
                    if pk >= dF_sd(iC)*xsd
                        true_peaks(idx1(iE)+ii-1,iC) = 1;
                        peak_ampl(idx1(iE)+ii-1,iC) = pk;
                        here2 = here(1)-round(samples_backw_in_sec*sf):here(end)+round(samples_forw_in_sec*sf);
                        try
                            act_bin(here2,iC) = 1;
                        catch
                            act_bin(here,iC) = 1;
                        end
                    else
                        true_fwd(idx1(iE),iC) = NaN;
                        true_rev(idx2(idx3),iC) = NaN;
                    end
                end
            catch
                % do nothing
            end
        end
    end
end

true_peaks(isnan(true_peaks)) = 0;
ttt = repmat(tt',[1 nCells]);
tmp = repmat([0:500],[numel(tt) 1]);

ev_thr = nan(size(true_peaks));
ev_thr(find(true_peaks)) = 1;

%% 
EV.events = ev_thr;
EV.peak_ampl = true_peaks;
EV.onset = true_fwd;
EV.offset = true_rev;
EV.bina = act_bin;



