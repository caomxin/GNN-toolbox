% Display a warning msg
function warn(num,msg)
global dynamicSystem

if num == 0
    warnmsg=[sprintf('WARNING:\t'), msg];
else
    warnmsg=['line ', num2str(num), sprintf(':\tWARNING: '), msg];
end
%on screen
disp(warnmsg);
%on file, if possible
if ~isempty(dynamicSystem) && isfield(dynamicSystem,'config') && isfield(dynamicSystem.config,'logFile')
    logfh = fopen(dynamicSystem.config.logFile,'a');
    if (dynamicSystem.config.useLogFile) && (logfh~=-1)
        c=clock;
        fprintf(logfh,['%2.0f:%2.0f:%2.0f> ' warnmsg '\n'],c(4),c(5),c(6));
    end
end
