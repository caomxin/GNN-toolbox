function [val,bool]=isInIntervalOC(str,min,max)
%isInIntervalOC(str,min,max) is true if num2str(str) is in (min,max]
value=str2double(str);
if isnan(value) 
    bool=false;
    val=NaN;
else
    bool=(value>min) && (value<=max);
    val=value;
end