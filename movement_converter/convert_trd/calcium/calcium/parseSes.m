function sesnames = parseSes(ses)
% PARSES MULTI-PART SESSION
%   sesnames = parseSes(ses)
%
%   <ses> session string input of format 'S1-S2-...-ACQ', e.g. 'S1-S2-S3-ACQ'
%   <sesnames> output with session name strings
%   
%   210324 SK V2, '-ACQ' 
%   210118 SK V1

idx = strfind(ses,'-');


% sesnames = struct;

if isempty(idx)
    disp('nothing to parse - no session separator in string');
    sesnames = ses;
elseif numel(idx) == 1
    sesnames = ses(1:idx-1);
elseif numel(idx) > 1
    here = [0 idx numel(ses)+1];
    cc = 1;
    for iS = 1:numel(here)-1
        this = ses(here(iS)+1:here(iS+1)-1);
        if ~contains(this,'ACQ')
            sesnames{cc,1} = this;
            if contains(ses,'ACQ')
                sesnames{cc,1} = strcat(sesnames{cc,1},'-ACQ');
            end
            cc = cc+1;
        end
    end
end

