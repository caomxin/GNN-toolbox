% Display an error msg
function err(num,msg)
global dynamicSystem
beep;
if num == 0
    errmsg=[sprintf('ERROR:\t'), msg];
else
    errmsg=['line ', num2str(num), sprintf(':\tERROR: '), msg];
end
%on screen
disp(errmsg);
%on file, if possible
if ~isempty(dynamicSystem) && isfield(dynamicSystem,'config') && isfield(dynamicSystem.config,'logFile')
    logfh = fopen(dynamicSystem.config.logFile,'a');
    if (dynamicSystem.config.useLogFile) && (logfh~=-1)
        c=clock;
        fprintf(logfh,['%2.0f:%2.0f:%2.0f> ' errmsg '\n'],c(4),c(5),c(6));
    end
end
