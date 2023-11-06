function SNR = calcSNR(infile,dF,act_bin,peak_ampl,saveFlag)
% CALCULATES SIGNAL-TO-NOISE RATIO
%   SNR = calcSNR(infile,dF,act_bin,peak_ampl,sf,saveFlag)
%   
%   <infile>    full path to MESc file
%   <dF>        calcium signal; dF/F (%) [samples x cells]
%   <act_bin>   binarised matrix of active periods [samples x cells]
%   <peak_ampl> peak amplitudes [dF/F] during active periods
%   <saveFlag>  save SNR to file [0/1]
%   <SNR>       signal-to-noise ratio
%   
%   adapted from HORSTO python code
%   
%   211001 SK V1


if nargin < 5
    saveFlag = 0;
end

% SNR = calcSNR(dF,decs,act_bin,peak_ampl)
% SNRwrapper(expt,decs)


%% defaults
peak_thr = 10; % min no of peaks
% sf = 31; % [fps]
% pre_win = 0.1; % [sec]
% post_win = 0.2; % [sec]

%% output paths/files
[filePath,expt,fileExt] = fileparts(infile)

outfile = strcat(filePath,'\suite2p\plane0\spks.npy');
outfile_snr = strcat(filePath,'\suite2p\plane0\snr.csv');

%% SNR calc
[~,nCells] = size(dF);

resp_win = act_bin; % activity, binarized
resp_win(resp_win == 0) = NaN;
noise_win = abs(act_bin-1); % noise window
noise_win(noise_win == 0) = NaN;

% spk_mean = nanmean(decs,1)'; % not used
% spk_sd = nanstd(decs,0,1)'; % not used

% peak_mean = nanmean(pk_ampl);
peak_num = sum(~isnan(peak_ampl),1)';

noise_mean = nanmean(dF.*noise_win,1)';
noise_median = nanmedian(dF.*noise_win,1)';
noise_sd = nanstd(dF.*noise_win,0,1)';
peak_mean = nanmean(dF.*resp_win,1)';
peak_sd = nanstd(dF.*resp_win,0,1)';
peak_mean_from_max = nanmean(peak_ampl)';
dF_mean = nanmean(dF,1)';

cell_snr = peak_mean./noise_sd;
cell_snr_peak = peak_mean_from_max./noise_sd;

%% export
% npy export
% try
%     % writeNPY(spks',outfile);
%     writeNPY(deconv',outfile);
% catch
%     disp('npy file export did not succeed')
% end
    
% csv export
var_names = {'idx' 'session_name' 'cell_id' 'spike_filter_id' 'noise_calc_id' 'snr_df_f' ... % same as in HORSTO code
    'snr_df_f_peak','noise_mean' 'noise_sd' 'dF_mean' 'peak_mean' 'peak_sd' 'peak_mean_from_max'}; % additional vars

cell_id = [0:nCells-1]';
cell_id = find(peak_num >= peak_thr)-1;

idx = [0:numel(cell_id)-1]';
ses_name = repmat(expt,numel(cell_id),1);
spk_filt = repmat('n/a',numel(cell_id),1);
noise_calc = repmat('n/a',numel(cell_id),1);

T = table(...
    idx,ses_name,cell_id,spk_filt,noise_calc,cell_snr(cell_id+1),...
    cell_snr_peak(cell_id+1),noise_mean(cell_id+1),noise_sd(cell_id+1),dF_mean(cell_id+1),peak_mean(cell_id+1),peak_sd(cell_id+1),peak_mean_from_max(cell_id+1),...
    'VariableNames',var_names);

if saveFlag
    writetable(T,outfile_snr,'Delimiter',',','WriteVariableNames',true,'FileType','text')
    disp('saved to csv file')
else
    disp('nothing has been save')
end

%% output
SNR.mean = cell_snr;
SNR.peak = cell_snr_peak;
SNR.table = T;

