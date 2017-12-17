function message1(msg)
global dynamicSystem

printScreen=1;

if ~isempty(dynamicSystem) && isfield(dynamicSystem,'config') && isfield(dynamicSystem.config,'logFile')
    if dynamicSystem.config.useLogFile
        logfh = fopen(dynamicSystem.config.logFile,'a');
        if (logfh==-1)
            dynamicSystem.config.useLogFile=0;
            warn(0, ['I can''t open log file <' dynamicSystem.config.logFile '>. I disable logging.']);
        else
            printScreen=0;
            c=clock;
            fprintf(logfh,['%2.0f:%2.0f:%2.0f> ' msg '\n'],c(4),c(5),c(6));
        end
    end
end
    
if (printScreen)
    msg=strrep(msg,'%%','%');   %to write in a file we must use %% for %, so here we need to replace back %% with %
    disp(msg)
end
    