function [dirs,filenames] = getDirs(expt,user)
% generate structure with default user-based data I/O directories used for any data handling
%   dirs = getDirs(expt,user)
%   <expt> needs to have a format like e.g. 'DON-123456_YYYYMMDD_INSTRUMENT_SESSION'
%   <user> user name, e.g. 'Fritz'
%   
%   210323 SK V4, edits and update
%   210114 SK V3, edits
%   210105 SK V2, bugfix
%   200414 SK V1

if nargin < 2
    user = getUser;
end
if nargin < 1
    expt = input('Specifiy experiment, e.g. DON-123456_YYYYMMDD_INSTRUMENT_SESSION : ','s');
end

[ms,dat,instr,ses] = parseExpt(expt);

compi = getenv('computername');
date = datestr(now,'yyyymmdd'); % fix

%% expt identifiers
% rigs
miniscope = '001P-I'; % inscopix miniscope (1-photon)
bscope_1p = '001P-T'; % Thorlabs B-scope (1-photon), technically a Basler cam
bscope_2p = '002P-T'; % Thorlabs B-scope (2-photon)
femtonics_1p = '001P-F'; % FemtoSmart (1-photon), technically a Basler cam
femtonics_2p = '002P-F'; % FemtoSmart (2-photon), mesc-files

% tracking
treadmill = 'TR-TRD'; % tracking treadmill
fearcond = '0000FC'; % tracking fear conditioning
basler = 'TR-BSL'; % tracking Basler, overhead arena
mousecam = '0000MC'; % tracking mousecam, side view

% histo
axioscan = '0000AS'; % Zeiss AxioScan.Z1
lsm = '000LSM'; % Zeiss LSM-700/800
spinsr = '0000SR'; % Olympus SpinSR

%% local drives
driveD = 'd:\';
driveE = 'e:\';
driveF = 'f:\';
driveU = 'u:\'; % mapped network dir; same as <uDir>

%% paths to donatolab network drives
% storage
groupDir = '\\biopz-jumbo.storage.p.unibas.ch\RG-FD02$\_Group\'; % aka, jumbo
scicoreDir = '\\scicore-puma.scicore.p.unibas.ch\BIOPZ-A-SCICORE-RG-FD02-Data01$\Data\'; % aka, puma
% scicoreDir = '';

% instruments
instrDir = '\\scicore-puma.p.unibas.ch\BIOPZ-A-SCICORE-RG-FD02-Data01$\Instruments'; 
nvistaDir = fullfile(instrDir,'nvista'); 
nvokeDir = fullfile(instrDir,'nvoke');
nvueDir = fullfile(instrDir,'nvue');

% collab
dunnDir = '\\scicore-puma.p.unibas.ch\BIOPZ-A-SCICORE-RG-FD02-Data01$\Dunnlab\';

% misc
uDir = '\\unibasel.ads.unibas.ch\BZ\';
vampDir = '\\vamp-fileserver.vamp.unibas.ch\vamp\';

%% new simple dirs
dirs.experiment = fullfile(scicoreDir,user,'Experiments',ms,dat,instr);
dirs.analysis = fullfile(scicoreDir,user,'Analysis',ms,dat,instr);
dirs.figures = fullfile(scicoreDir,user,'Figures',ms,dat,instr);

%% storage dirs etc.
dirs.scratch.local = 'd:\scratch\';
dirs.scratch.network = '';

dirs.storage.group = fullfile(groupDir,user); % BZ
dirs.storage.scicore = fullfile(scicoreDir,user); % sciCORE

dirs.fileservers.u = fullfile(uDir);
dirs.fileservers.vamp = fullfile(vampDir);

dirs.instruments.nvista = nvistaDir;
dirs.instruments.nvoke = nvokeDir;
dirs.instruments.nvue = nvueDir;

dirs.collab.dunnlab = dunnDir;

%% resources, code mainly
dirs.resources.code.matlab = fullfile(groupDir,'Resources','Code','mfiles');
dirs.resources.code.python = fullfile(groupDir,'Resources','Code','python');
dirs.resources.code.vr = fullfile(groupDir,'Resources','Code','bonsai');
dirs.resources.code.deeplabcut = '';
dirs.resources.code.suite2p = '';
dirs.resources.code.arduino = '';

%% imageJ
dirs.imagej.local = 'c:\Program Files (x86)\ImageJ\';
dirs.imagej.network = '';

dirs.fiji.local = '';
dirs.fiji.network = '';

%% experiment dirs (raw data)
dirs.full.experiment.imaging.p.miniscope = fullfile(scicoreDir,user,'Experiments',ms,dat,miniscope); % inscopix miniscope (1-photon)
dirs.full.experiment.imaging.p.bscope = fullfile(scicoreDir,user,'Experiments',ms,dat,bscope_1p); % Thorlabs B-scope (1-photon), technically a Basler cam
dirs.full.experiment.imaging.p.femtonics = fullfile(scicoreDir,user,'Experiments',ms,dat,femtonics_1p); % FemtoSmart (1-photon), technically a Basler cam
dirs.full.experiment.imaging.pp.bscope = fullfile(scicoreDir,user,'Experiments',ms,dat,bscope_2p); % Thorlabs B-scope (2-photon)
dirs.full.experiment.imaging.pp.bscope_tif = fullfile(scicoreDir,user,'Experiments',ms,dat,bscope_2p,'tif'); % B-scope tiff-conversion
dirs.full.experiment.imaging.pp.femtonics = fullfile(scicoreDir,user,'Experiments',ms,dat,femtonics_2p); % FemtoSmart (2-photon), mesc-files
dirs.full.experiment.imaging.pp.femtonics_tif = fullfile(scicoreDir,user,'Experiments',ms,dat,femtonics_2p,'tif'); % FemtoSmart tiff-conversion

dirs.full.experiment.tracking.treadmill = fullfile(scicoreDir,user,'Experiments',ms,dat,treadmill); % tracking treadmill
dirs.full.experiment.tracking.fearcond = fullfile(scicoreDir,user,'Experiments',ms,dat,fearcond); % tracking fear conditioning
dirs.full.experiment.tracking.basler = fullfile(scicoreDir,user,'Experiments',ms,dat,basler); % tracking Basler, overhead arena
dirs.full.experiment.tracking.mousecam = fullfile(scicoreDir,user,'Experiments',ms,dat,mousecam); % tracking mousecam, side view

dirs.full.experiment.microscopy.axioscan = fullfile(scicoreDir,user,'Experiments',ms,dat,axioscan); % Zeiss AxioScan.Z1
dirs.full.experiment.microscopy.lsm = fullfile(scicoreDir,user,'Experiments',ms,dat,lsm); % Zeiss LSM-800
dirs.full.experiment.microscopy.spinsr = fullfile(scicoreDir,user,'Experiments',ms,dat,spinsr); % Olympus SpinSR

%% analysis dirs (derived data)
dirs.full.analysis.imaging.p.miniscope = fullfile(scicoreDir,user,'Analysis',ms,dat,miniscope);
dirs.full.analysis.imaging.p.bscope = fullfile(scicoreDir,user,'Analysis',ms,dat,bscope_1p);
dirs.full.analysis.imaging.p.femtonics = fullfile(scicoreDir,user,'Analysis',ms,dat,femtonics_1p);
dirs.full.analysis.imaging.pp.bscope = fullfile(scicoreDir,user,'Analysis',ms,dat,bscope_2p);
dirs.full.analysis.imaging.pp.bscope_s2p = fullfile(scicoreDir,user,'Analysis',ms,dat,bscope_2p,'suite2p');
dirs.full.analysis.imaging.pp.bscope_s2p_alt = fullfile(scicoreDir,user,'Analysis',ms,dat,bscope_2p,'tif','suite2p');
dirs.full.analysis.imaging.pp.femtonics = fullfile(scicoreDir,user,'Analysis',ms,dat,femtonics_2p);
dirs.full.analysis.imaging.pp.femtonics_s2p = fullfile(scicoreDir,user,'Analysis',ms,dat,femtonics_2p,'suite2p');
dirs.full.analysis.imaging.pp.femtonics_s2p_alt = fullfile(scicoreDir,user,'Analysis',ms,dat,femtonics_2p,'tif','suite2p');

dirs.full.analysis.tracking.treadmill = fullfile(scicoreDir,user,'Analysis',ms,dat,treadmill);
dirs.full.analysis.tracking.fearcond = fullfile(scicoreDir,user,'Analysis',ms,dat,fearcond);
dirs.full.analysis.tracking.basler = fullfile(scicoreDir,user,'Analysis',ms,dat,basler);
dirs.full.analysis.tracking.basler_dlc = fullfile(scicoreDir,user,'Analysis',ms,dat,basler,'dlc');
dirs.full.analysis.tracking.basler_dlc_alt = fullfile(scicoreDir,user,'Analysis',ms,dat,basler,'000DLC');
dirs.full.analysis.tracking.mousecam = fullfile(scicoreDir,user,'Analysis',ms,dat,mousecam);
dirs.full.analysis.tracking.mousecam_dlc = fullfile(scicoreDir,user,'Analysis',ms,dat,mousecam,'dlc');
dirs.full.analysis.tracking.mousecam_dlc_alt = fullfile(scicoreDir,user,'Analysis',ms,dat,mousecam,'000DLC');

dirs.full.analysis.microscopy.axioscan = fullfile(scicoreDir,user,'Analysis',ms,dat,axioscan);
dirs.full.analysis.microscopy.lsm = fullfile(scicoreDir,user,'Analysis',ms,dat,lsm);
dirs.full.analysis.microscopy.spinsr = fullfile(scicoreDir,user,'Analysis',ms,dat,spinsr);

%% figure dirs (figs)
dirs.full.figures.imaging.p.miniscope = fullfile(scicoreDir,user,'Figures',ms,dat,miniscope);
dirs.full.figures.imaging.p.bscope = fullfile(scicoreDir,user,'Figures',ms,dat,bscope_1p);
dirs.full.figures.imaging.p.femtonics = fullfile(scicoreDir,user,'Figures',ms,dat,femtonics_1p);
dirs.full.figures.imaging.pp.bscope = fullfile(scicoreDir,user,'Figures',ms,dat,bscope_2p);
dirs.full.figures.imaging.pp.femtonics = fullfile(scicoreDir,user,'Figures',ms,dat,femtonics_2p);

dirs.full.figures.tracking.treadmill = fullfile(scicoreDir,user,'Figures',expt,treadmill);
dirs.full.figures.tracking.fearcond = fullfile(scicoreDir,user,'Figures',expt,fearcond);
dirs.full.figures.tracking.basler = fullfile(scicoreDir,user,'Figures',expt,basler);
dirs.full.figures.tracking.mousecam = fullfile(scicoreDir,user,'Figures',expt,mousecam);

dirs.full.figures.microscopy.axioscan = fullfile(scicoreDir,user,'Figures',expt,axioscan);
dirs.full.figures.microscopy.lsm = fullfile(scicoreDir,user,'Figures',expt,lsm);
dirs.full.figures.microscopy.spinsr = fullfile(scicoreDir,user,'Figures',expt,spinsr);

%% filenames
[ms,dat,instr,ses] = parseExpt(expt);
sesnames = parseSes(ses);

filenames.miniscope = strcat(expt,'.isxd');
filenames.bscope_1p = '?';
filenames.bscope_2p = '?';
filenames.femtonics_1p = '?';
filenames.femtonics_2p = strcat(expt,'.mesc');

if numel(sesnames) > 1
    for iS = 1:numel(sesnames)
        filenames.treadmill{iS,1} = strcat(ms,'_',dat,'_',treadmill,'_',sesnames{iS},'.mat');
        filenames.mousecam{iS,1} = strcat(ms,'_',dat,'_',mousecam,'_',sesnames{iS},'.mp4');
        filenames.basler{iS,1} = strcat(ms,'_',dat,'_',basler,'_',sesnames{iS},'.mp4');
    end
else
    filenames.treadmill = strcat(ms,'_',dat,'_',treadmill,'_',sesnames{iS},'.mat');
    filenames.mousecam = strcat(ms,'_',dat,'_',mousecam,'_',sesnames{iS},'.mp4');
    filenames.basler = strcat(ms,'_',dat,'_',basler,'_',sesnames{iS},'.mp4');
end

filenames.fearcond = '?';
