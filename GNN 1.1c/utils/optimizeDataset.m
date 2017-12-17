function optimizeDataset(set)
global dataSet dynamicSystem
dataSet.(set).connMatrix=logical(dataSet.(set).connMatrix);
if ~dynamicSystem.config.useBalancedDataset
%     if (size(dataSet.trainSet.targets,1)==1)
        dataSet.(set).maskMatrix=spones(dataSet.(set).maskMatrix);
        dataSet.(set).maskMatrix=logical(dataSet.(set).maskMatrix);
%     end
end