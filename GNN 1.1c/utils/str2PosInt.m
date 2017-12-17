function [val,bool]=str2PosInt(str)
%[val,bool]=str2PosInt(str). bool is true if str is a string containing a
%positive integer (and val is the value). Else bool = false
if ~isempty(findstr(str,'.')) 
    bool=false;
    val=NaN;
else
    value = str2double(str);
    if isnan(value) || value<=0
        bool=false;
        val=NaN;
    else
        bool=true;
        val=value;
    end
end
