function res=isInIntervalCC(value,min,max)
%isisInIntervalCC(value,min,max) is true if value is in [min,max]
if isnan(value) || ~isnumeric(value)
    res=false;
else
    res=(value>=min) && (value<=max);
end
