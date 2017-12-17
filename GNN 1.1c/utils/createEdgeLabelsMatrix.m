function createEdgeLabelsMatrix

global dataSet

%  index=find(dataSet.trainSet.connMatrix);
%  sx=size(index,1);
[r,c]=find(dataSet.trainSet.connMatrix);
sx=size(r,1);

if ~isfield(dataSet.trainSet,'edges')
    error('I can''t find edge labels in the train set');
end
dataSet.trainSet.edgeLabels=ones(dataSet.config.edgeLabelsDim,sx);  
for i=1:size(dataSet.trainSet.edges,2)
%      ind=find(index==(dataSet.trainSet.edges(i).father-1)*dataSet.trainSet.nNodes+dataSet.trainSet.edges(i).child);
    ind=find((c==(dataSet.trainSet.edges(i).father))&(r==(dataSet.trainSet.edges(i).child)));
    if isempty(ind)
        error('I''ve found a label for an arc in the train set that doesn''t seem to exist');
    end
    dataSet.trainSet.edgeLabels(:,ind)=dataSet.trainSet.edges(i).value;
end


% index=find(dataSet.validationSet.connMatrix);
% sx=size(index,1);
[r,c]=find(dataSet.validationSet.connMatrix);
sx=size(r,1);

if ~isfield(dataSet.validationSet,'edges')
    error('I can''t find edge labels in the validation set');
end
dataSet.validationSet.edgeLabels=ones(dataSet.config.edgeLabelsDim,sx);  
for i=1:size(dataSet.validationSet.edges,2)
%     ind=find(index==(dataSet.validationSet.edges(i).father-1)*dataSet.validationSet.nNodes+dataSet.validationSet.edges(i).child);
    ind=find((c==(dataSet.validationSet.edges(i).father))&(r==(dataSet.validationSet.edges(i).child)));
    if isempty(ind)
        error('I''ve found a label for an arc in the validation set that doesn''t seem to exist');
    end
    dataSet.validationSet.edgeLabels(:,ind)=dataSet.validationSet.edges(i).value;
end


% index=find(dataSet.testSet.connMatrix);
% sx=size(index,1);
[r,c]=find(dataSet.testSet.connMatrix);
sx=size(r,1);

if ~isfield(dataSet.testSet,'edges')
    error('I can''t find edge labels in the test set');
end
dataSet.testSet.edgeLabels=ones(dataSet.config.edgeLabelsDim,sx);  
for i=1:size(dataSet.testSet.edges,2)
%     ind=find(index==(dataSet.testSet.edges(i).father-1)*dataSet.testSet.nNodes+dataSet.testSet.edges(i).child);
    ind=find((c==(dataSet.testSet.edges(i).father))&(r==(dataSet.testSet.edges(i).child)));
    if isempty(ind)
        error('I''ve found a label for an arc in the test set that doesn''t seem to exist');
    end
    dataSet.testSet.edgeLabels(:,ind)=dataSet.testSet.edges(i).value;
end

dataSet.trainSet=rmfield(dataSet.trainSet,'edges');
dataSet.validationSet=rmfield(dataSet.validationSet,'edges');
dataSet.testSet=rmfield(dataSet.testSet,'edges');