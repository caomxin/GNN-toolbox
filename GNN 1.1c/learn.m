function learn
% this function does the following:
%
%   1. performs a test to assure the system has been configured properly
%   2. calls the main learning function <learn_.m>


global dynamicSystem learning dataSet VisualMode
if (isempty(dynamicSystem) || ~isfield(dynamicSystem,'config') || isempty(dynamicSystem.config)  ||...
        ~isfield(dynamicSystem.config,'configured') || dynamicSystem.config.configured~=1)
    err(0,sprintf('The system seems misconfigured.\nRun Configure or ConfigurationTool'));
    return
end

learn_;
