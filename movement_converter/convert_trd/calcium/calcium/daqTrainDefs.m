function dtdefs = daqTrainDefs
% daqTrainDefs loads default params for DAQTRAINEVAL
%   
%   200505 SK

% if nargin < 1
%     metadata = [];
% end

%% standard defs
beltLen = 150; % [cm]
sf = 10000; % [Hz], MCC freq

ppr = 500; % encoder pulses per rev
whDiam = 10; % [cm], wheel diam.
% thrTTL = [1 4]; % [V], thresh for TTL high/low
% thrLap = .4; % THIS NEEDS TO BE TESTED; was .75
thrTTL = 3.5;
thrLap = thrTTL;
disp('warning: uses TTL threshold for rotary encoder & lap detector -- 200923 SK')
thrGalvo = 2.5; % [V]
thrEnc = .15; % +/- pulse-per-lap threshold -> [0.95 1.05] of mean ppl
dpp = (whDiam*pi)/ppr; % dist per pulse [cm]

encInv = 1; % encoder inverted
lp = .1; % low pass
lp_lap = lp/10; % for lap signal 
sd_lap = 7.5; % n-times SD threshold for detection on diff

thrLocom = 2; % [cm/s]
thrStill = 1; % [s]

%%
dtdefs.beltLen = beltLen;
dtdefs.sf = sf;
dtdefs.wheel.ppr = ppr;
dtdefs.wheel.diam = whDiam;
dtdefs.wheel.dpp = dpp;
dtdefs.thr.TTL = thrTTL;
dtdefs.thr.galvo = thrGalvo;
dtdefs.thr.lap = thrLap;
dtdefs.thr.run = thrLocom;
dtdefs.thr.pause = thrStill; 
dtdefs.thr.encoder = thrEnc;

dtdefs.misc.invert = encInv;
dtdefs.misc.lowpass = lp;
dtdefs.misc.lowpass_lap = lp_lap;
dtdefs.misc.sd_lap = sd_lap;

disp('Default parameters for DAQ TRAINING loaded')

