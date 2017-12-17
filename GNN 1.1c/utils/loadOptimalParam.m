function loadOptimalParam(file)
% a = pwd;
% cd /tmp
if nargin == 0
    load('/tmp/optimalParam.mat');
else
    load(file);
end
% cd(a);

global learning dynamicSystem


learning.current.optimalParameters=optimalParam;
learning.config=learnconfig;
dynamicSystem.config = sysconfig;
