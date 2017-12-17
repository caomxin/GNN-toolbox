function test
% test function

% A classification problem and a regression problem need two different testing strategies. 
% The function choices the one needed from dataSet.config.type

global dataSet learning dynamicSystem
if isempty(dataSet)
    err(0,'Cannot find the datasets')
    return
elseif ~isfield(learning,'current')
    err(0,'Cannot find learning data')
    return
end
if ~isfield(dataSet,'testSet')
    err(0, 'Cannot find the testSet')
    return
end
if ~isfield(dataSet.testSet,'neuralModel')
    prepareDataset('testSet');
    optimizeDataset('testSet');
end
    

if strcmp(func2str(dynamicSystem.config.computeErrorFunction),'neuralModelQuadraticComputeError')
    test4uniform;
elseif strcmp(func2str(dynamicSystem.config.computeErrorFunction),'autoassociatorComputeError')
    test4autoassociator;
elseif strcmp(dataSet.config.type,'classification')
    test4classification;
elseif strcmp(dataSet.config.type,'regression')
    test4regression;
end
