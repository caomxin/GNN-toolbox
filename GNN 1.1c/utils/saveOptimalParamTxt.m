function saveOptimalParam(file)

global learning dynamicSystem


optimalParam = learning.current.optimalParameters;
sysconfig = dynamicSystem.config;
learnconfig = learning.config;
a = pwd;
cd /tmp

if nargin == 0
    file='/tmp/optimalParam.txt';
end


h=fopen(file,'wt');
if h==-1
    error(['Cannot open file ' file]);
else
    fprintf(h,'#nConfiguration: nStates useLabelledEdges errorFunction \n');
    fprintf(h,'#transitionNet: architecture(#in, #h, #o) outActivationFunction \n');
    fprintf(h,'#outNet: architecture(#in, #h, #o) outActivationFunction \n');
    fprintf(h,'#OptimalParam: outNet(w1,b1,w2,b2) transitionNet(w1,b1,w2,b2) \n');
    
    fprintf(h,'%d %d %s\n',sysconfig.nStates,sysconfig.useLabelledEdges,'mse');
    fprintf(h,'%d %d %d %s\n',sysconfig.transitionNet.nInputs,sysconfig.transitionNet.nHiddens,sysconfig.transitionNet.nOuts, sysconfig.transitionNet.outActivationType);
    fprintf(h,'%d %d %d %s\n',sysconfig.outNet.nInputs,sysconfig.outNet.nHiddens,sysconfig.outNet.nOuts, sysconfig.outNet.outActivationType);
    
    fprintf(h,'%d %d\n',size(optimalParam.outNet.weights1));
    fprintf(h, '%12.8e ', optimalParam.outNet.weights1);
    fprintf(h,'\n%d %d\n',size(optimalParam.outNet.bias1));
    fprintf(h, '%12.8e ', optimalParam.outNet.bias1);
    
    fprintf(h,'\n%d %d\n',size(optimalParam.outNet.weights2));
    fprintf(h, '%12.8e ', optimalParam.outNet.weights2);
    fprintf(h,'\n%d %d\n',size(optimalParam.outNet.bias2));
    fprintf(h, '%12.8e ', optimalParam.outNet.bias2);
    
    fprintf(h,'\n%d %d\n',size(optimalParam.transitionNet.weights1));
    fprintf(h, '%12.8e ', optimalParam.transitionNet.weights1);
    fprintf(h,'\n%d %d\n',size(optimalParam.transitionNet.bias1));
    fprintf(h, '%12.8e ', optimalParam.transitionNet.bias1);
    
    fprintf(h,'\n%d %d\n',size(optimalParam.transitionNet.weights2));
    fprintf(h, '%12.8e ', optimalParam.transitionNet.weights2);
    fprintf(h,'\n%d %d\n',size(optimalParam.transitionNet.bias2));
    fprintf(h, '%12.8e ', optimalParam.transitionNet.bias2);
    
    fprintf(h,'\n');
    fclose(h);
    
    
    
end
cd(a)
clear a optimalParam sysconfig