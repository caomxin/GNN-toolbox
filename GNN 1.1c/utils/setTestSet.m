function setTestSet(graphNum,nNodes,connMatrix,maskVector,nodeLabels,edgeLabels,targets)

global dataSet dynamicSystem

dataSet.testSet=[];


dataSet.testSet.graphNum=graphNum;
dataSet.testSet.nNodes=nNodes;

dataSet.testSet.connMatrix=connMatrix;
sz=size(maskVector,1);
dataSet.testSet.maskMatrix=sparse(1:sz,1:sz,maskVector,sz,sz);
dataSet.testSet.nodeLabels=nodeLabels;
dataSet.testSet.edgeLabels=edgeLabels;
dataSet.testSet.targets=targets;

%% evaluating again useful data
E=speye(dynamicSystem.config.nStates);

[dataSet.testSet.neuralModel.childOfArc,dataSet.testSet.neuralModel.fatherOfArc]=find(dataSet.testSet.connMatrix);
dataSet.testSet.nArcs=size(dataSet.testSet.neuralModel.childOfArc,1);
dataSet.testSet.neuralModel.fatherToArcMatrix=logical(kron(sparse(1:dataSet.testSet.nArcs,...
    dataSet.testSet.neuralModel.fatherOfArc,ones(dataSet.testSet.nArcs,1),dataSet.testSet.nArcs,dataSet.testSet.nNodes),E));
dataSet.testSet.neuralModel.childToArcMatrix=logical(kron(sparse(1:dataSet.testSet.nArcs,...
    dataSet.testSet.neuralModel.childOfArc,ones(dataSet.testSet.nArcs,1),dataSet.testSet.nArcs,dataSet.testSet.nNodes),E));
[r,c]=find(ones(dynamicSystem.config.nStates,dynamicSystem.config.nStates*dataSet.testSet.nArcs,'uint8'));
dataSet.testSet.neuralModel.toBlockMatrixColumn=c;
dataSet.testSet.neuralModel.toBlockMatrixRow=r+floor((c-1)/dynamicSystem.config.nStates)*dynamicSystem.config.nStates;
dataSet.testSet.neuralModel.arcOfFatherMatrix=logical(sparse(1:dataSet.testSet.nArcs, dataSet.testSet.neuralModel.fatherOfArc,...
    ones(dataSet.testSet.nArcs,1),dataSet.testSet.nArcs,dataSet.testSet.nNodes));
