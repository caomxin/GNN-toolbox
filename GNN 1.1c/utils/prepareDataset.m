function prepareDataset(set)
%This function prepare the dataset evaluation .neuralModel part and
%optimizing
global dataSet dynamicSystem
if strcmp(dynamicSystem.config.type,'neural')
    % message1(['Preparing the ' set '...'])
    E=speye(dynamicSystem.config.nStates);
    [dataSet.(set).neuralModel.childOfArc,dataSet.(set).neuralModel.fatherOfArc]=find(dataSet.(set).connMatrix);
    dataSet.(set).nArcs=size(dataSet.(set).neuralModel.childOfArc,1);
    dataSet.(set).neuralModel.fatherToArcMatrix=logical(kron(sparse(1:dataSet.(set).nArcs,...
        dataSet.(set).neuralModel.fatherOfArc,ones(dataSet.(set).nArcs,1),dataSet.(set).nArcs,dataSet.(set).nNodes),E));
    dataSet.(set).neuralModel.childToArcMatrix=logical(kron(sparse(1:dataSet.(set).nArcs,...
        dataSet.(set).neuralModel.childOfArc,ones(dataSet.(set).nArcs,1),dataSet.(set).nArcs,dataSet.(set).nNodes),E));
    [r,c]=find(ones(dynamicSystem.config.nStates,dynamicSystem.config.nStates*dataSet.(set).nArcs,'uint8'));
    dataSet.(set).neuralModel.toBlockMatrixColumn=c;
    dataSet.(set).neuralModel.toBlockMatrixRow=r+floor((c-1)/dynamicSystem.config.nStates)*dynamicSystem.config.nStates;
    dataSet.(set).neuralModel.arcOfFatherMatrix=logical(sparse(1:dataSet.(set).nArcs, dataSet.(set).neuralModel.fatherOfArc,...
        ones(dataSet.(set).nArcs,1),dataSet.(set).nArcs,dataSet.(set).nNodes));
end
