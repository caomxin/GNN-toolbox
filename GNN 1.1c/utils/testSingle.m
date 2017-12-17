function outs=testSingle()

global dataSet dynamicSystem learning
%returns GNN outputs on the testSet using optimalParameters

dataSet.testSet.forwardSteps=30;

% evaluating optimal parameters on testSet
[x,testForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),...
    'testSet',1);
[error,testOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,1);

outs=testOutState.outNetState.outs;

