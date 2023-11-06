function [ms,dat,instr,ses,sesnames] = parseExpt(expt)
% PARSES EXPERIMENT STRINGS
%   [ms,dat,instr,ses] = parseExpt(expt)
%
%   <expt> experiment string input of format 'mouse_date_instrument_session', e.g. 'DON-12345_20210123_002P-F_S1-ACQ'
%   <ms> <dat> <instr> <ses> are the output strings
%   
%   210114 SK V1

idx = strfind(expt,'_');

if numel(idx) ~= 3
    error('expt string format unknown');
end

ms = expt(1:idx(1)-1);
dat = expt(idx(1)+1:idx(2)-1);
instr = expt(idx(2)+1:idx(3)-1);
ses = expt(idx(3)+1:end);

sesnames = parseSes(ses);
