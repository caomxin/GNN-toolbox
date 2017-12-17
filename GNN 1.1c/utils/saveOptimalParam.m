function saveOptimalParam(file)

global learning dynamicSystem

optimalParam = learning.current.optimalParameters;
sysconfig = dynamicSystem.config;
learnconfig = learning.config;
a = pwd;
cd /tmp

if nargin == 0
    save('/tmp/optimalParam.mat','optimalParam','sysconfig','learnconfig')
else
    save(file,'optimalParam','sysconfig','learnconfig')
end
cd(a)
