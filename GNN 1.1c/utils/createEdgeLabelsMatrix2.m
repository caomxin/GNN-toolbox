function data=createEdgeLabelsMatrix2(data)

index=find(data.trainSet.connMatrix);
sx=size(index,1);

if ~isfield(data.trainSet,'edges')
    error('I can''t find edge labels in the train set');
end
data.trainSet.edgeLabels=ones(data.config.edgeLabelsDim,sx);  
for i=1:size(data.trainSet.edges,2)
    ind=find(index==(data.trainSet.edges(i).father-1)*data.trainSet.nNodes+data.trainSet.edges(i).child);
    if isempty(ind)
        error('I''ve found a label for an arc in the train set that doesn''t seem to exist');
    end
    data.trainSet.edgeLabels(:,ind)=data.trainSet.edges(i).value;
end


index=find(data.validationSet.connMatrix);
sx=size(index,1);

if ~isfield(data.validationSet,'edges')
    error('I can''t find edge labels in the validation set');
end
data.validationSet.edgeLabels=ones(data.config.edgeLabelsDim,sx);  
for i=1:size(data.validationSet.edges,2)
    ind=find(index==(data.validationSet.edges(i).father-1)*data.validationSet.nNodes+data.validationSet.edges(i).child);
    if isempty(ind)
        error('I''ve found a label for an arc in the validation set that doesn''t seem to exist');
    end
    data.validationSet.edgeLabels(:,ind)=data.validationSet.edges(i).value;
end

index=find(data.testSet.connMatrix);
sx=size(index,1);

if ~isfield(data.testSet,'edges')
    error('I can''t find edge labels in the test set');
end
data.testSet.edgeLabels=ones(data.config.edgeLabelsDim,sx);  
for i=1:size(data.testSet.edges,2)
    ind=find(index==(data.testSet.edges(i).father-1)*data.testSet.nNodes+data.testSet.edges(i).child);
    if isempty(ind)
        error('I''ve found a label for an arc in the test set that doesn''t seem to exist');
    end
    data.testSet.edgeLabels(:,ind)=data.testSet.edges(i).value;
end

data.trainSet=rmfield(data.trainSet,'edges');
data.validationSet=rmfield(data.validationSet,'edges');
data.testSet=rmfield(data.testSet,'edges');