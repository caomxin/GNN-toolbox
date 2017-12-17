

global dataSet dynamicSystem
dataSet.config.type='classification';



dataSet.config.nodeLabelsDim=1;
dataSet.config.edgeLabelsDim=1;
dataSet.config.rejectUpperThreshold=0;
dataSet.config.rejectLowerThreshold=0;


dataSet.trainSet.nNodes=20;
dataSet.trainSet.connMatrix=sparse(ones(20)-eye(20));
dataSet.trainSet.maskMatrix=sparse(eye(20));
dataSet.trainSet.nodeLabels=ones(1,20);
dataSet.trainSet.targets=-ones(1,20);
dataSet.trainSet.targets(1,1:5)=1;
dataSet.trainSet.edges=[];
k=1;
for i=1:5
    for j=1:5
        if j~=i
            dataSet.trainSet.edges(k).father=i;
            dataSet.trainSet.edges(k).child=j;
            dataSet.trainSet.edges(k).value=5;
            k=k+1;
        end
    end
end



dataSet.validationSet.nNodes=20;
dataSet.validationSet.connMatrix=sparse(ones(20)-eye(20));
dataSet.validationSet.maskMatrix=sparse(eye(20));
dataSet.validationSet.nodeLabels=ones(1,20);
dataSet.validationSet.targets=-ones(1,20);
dataSet.validationSet.targets(1,5:8)=1;
dataSet.validationSet.edges=[];
k=1;
for i=5:8
    for j=5:8
        if j~=i
            dataSet.validationSet.edges(k).father=i;
            dataSet.validationSet.edges(k).child=j;
            dataSet.validationSet.edges(k).value=5;
            k=k+1;
        end
    end
end




dataSet.testSet.nNodes=20;
dataSet.testSet.connMatrix=sparse(ones(20)-eye(20));
dataSet.testSet.maskMatrix=sparse(eye(20));
dataSet.testSet.nodeLabels=ones(1,20);
dataSet.testSet.targets=-ones(1,20);
dataSet.testSet.targets(1,3:8)=1;
dataSet.testSet.edges=[];
k=1;
for i=3:8
    for j=3:8
        if j~=i
            dataSet.testSet.edges(k).father=i;
            dataSet.testSet.edges(k).child=j;
            dataSet.testSet.edges(k).value=5;
            k=k+1;
        end
    end
end

dataSet.config.useLabelledEdges=1;

createEdgeLabelsMatrix;