function evaluateAccuracyOnGraphs(set,param)
global dataSet testing
% evaluate accuracy on graphs (instead of on nodes)
if isfield(dataSet.(set),'graphNum')&&isfield(dataSet.config,'graphDim')
    if dataSet.(set).graphNum > 1
        % it makes sense to evaluate accuracy on graphs
        eval(['err=zeros(1,size(dataSet.' (set) '.targets,2));']);
        eval(['err(testing.' (param) '.' (set) '.mistakenPatternIndex)=1;']);
        eval(['g_err=reshape(err,dataSet.config.graphDim,dataSet.' (set) '.graphNum);']);
        eval(['testing.' (param) '.' (set) '.accuracyOnGraphs=size(find(sum(g_err)==0),2)/dataSet.' (set) '.graphNum;']);
    end
end
