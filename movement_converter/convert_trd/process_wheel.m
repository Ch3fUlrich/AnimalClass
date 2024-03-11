function process_wheel(filename)
%addpath("calcium/calcium/")
%addpath("npy-matlab\")
wheel = load(filename);

cha = wheel.trainingdata(:,1);
chb = wheel.trainingdata(:,2);
cha(cha<2.5)=0;
cha(cha>=2.5)=1;
chb(chb<2.5)=0;
chb(chb>=2.5)=1;
out = quadenc(cha, chb);

[filepath, fname, ext] = fileparts(filename);

session_part = regexp(fname, 'S[0-9]', "match"){1};
wheel_name = strcat(session_part, '_wheel.npy');
wheel_fname = strcat(filepath, "/" ,wheel_name);
writeNPY(out, wheel_fname);

chc = wheel.trainingdata(:,4);
galvo_trigger_name = strcat(session_part, '_2p_galvo_trigger.npy');
galvo_fname = strcat(filepath, "/" , galvo_trigger_name);
writeNPY(chc, galvo_fname)