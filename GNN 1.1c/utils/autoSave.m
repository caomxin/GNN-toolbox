function autoSave

global dynamicSystem dataSet learning testing

if nargin==0
    fname=['autoSave/' dynamicSystem.config.name '.mat'];
end
try
    saveAll(fname);
    message1([sprintf('\t\t\t') 'Autosaving completed']);
catch
    [errstr, errid] = lasterr;
    warn(0,[sprintf('\t\t\t') 'Autosaving failed! Autosaving will be disabled' sprintf('\n\t') strrep(errstr,sprintf('\n'),sprintf('\t'))]);
    dynamicSystem.config.useAutoSave=0;
end
