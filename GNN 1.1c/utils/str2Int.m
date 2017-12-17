function [val,bool]=str2Int(str)
%[val,bool]=str2Int(str). bool is true if str is a string containing an
% integer (and val is the value). Else bool = false
if ~isempty(findstr(str,',')) || ~isempty(findstr(str,'.')) 
    bool=false;
else
    value = str2double(str);
    if isnan(value)
        bool=false;
        val=NaN
    else
        bool=true;
        val=value;
    end
end
