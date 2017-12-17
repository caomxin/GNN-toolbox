% The function build a dataset composed by random graphs:
%
%   1. it generates a random subgraph S to be recognized
%   2. it generates a set of larger graphs (each one having the same dimension)
%   3. it inserts S into any graph
%   4. it adds noise to the labels of each graph
%   5. it attaches targets to each node: the target is 
%           +1 if the node belongs to the subgraph 
%           -1 otherwise
%   6. the dataset is subdivided in a train set, a validation set and a test set

function makeSubGraphDataset(file)
% Create a dataset for the subgraph recognition problem
% USAGE: makeSubGraphDataset(<file.config>)

global dataSet params params_name
if ~isempty(dataSet)
    dataSet=[];
end


% load parameters from file
try
    if nargin ~= 1
        file='SubGraphMatchingDataset.config';     %default config file
    end
    [fid,message] = fopen(file,'rt');
    if fid == -1
        err(0,['<' file '>: ' message]);
    end
    
    params = [0 0 0 0 0 0 0 0 0 0 0 0 0];
    % subMatDim graphDim maxLabel nodeLabelsDim graphDensity normalNoiseCoefficient uniformNoiseCoefficient noiseMean
    % rejectUpperThreshold rejectLowerThreshold trainSet.graphNum validationSet.graphNum testSet.graphNum
    params_name={'subMatDim' 'graphDim' 'maxLabel' 'nodeLabelsDim' 'graphDensity' 'normalNoiseCoefficient'...
            'uniformNoiseCoefficient' 'noiseMean' 'rejectUpperThreshold' 'rejectLowerThreshold' 'trainSet.graphNum'...
            'validationSet.graphNum' 'testSet.graphNum'};
    
    delimiter = [';' ' ' 9 13];         %ASCII(9) = TABS, ASCII(13) = carriage return
    numline=0;
    while feof(fid) == 0
        numline=numline+1;
        line = fgetl(fid);
        if isempty(line)
            warn(numline, 'line is empty. Please remove it or comment it');
        elseif line(1)=='#'
            %this is a comment line
            continue;
        else    
            line = strtok(line, delimiter);
            k = strfind(line,'=');
            if size(k,2)==0
                err(numline,'No ''='' detected');
            elseif (size(k,2)>1)
                err(numline,'More than one "=" detected');
            end
            name = line(1:k-1);
            value = line(k+1:end);
            check_valueok(name,value,num2str(numline));            
        end
    end
    fclose(fid);
    check_thereisall;
catch
    fclose all;
    return;
end

dataSet.config.type='classification';
dataSet.config.centralThreshold=((dataSet.config.graphDim-dataSet.config.subMatDim)-...
    dataSet.config.subMatDim)/dataSet.config.graphDim;

%GRAPH GENERATION START
tmp.graphNum=dataSet.trainSet.graphNum+dataSet.testSet.graphNum+dataSet.validationSet.graphNum;
tmp.nNodes=tmp.graphNum*dataSet.config.graphDim;
tmp.maskMatrix=speye(tmp.nNodes);


tmp.nodeLabelsDim=dataSet.config.nodeLabelsDim;
tmp.subMatDim=dataSet.config.subMatDim;
tmp.maxLabel=dataSet.config.maxLabel;
tmp.graphDim=dataSet.config.graphDim;
tmp.connMatrix=sparse(tmp.nNodes,tmp.nNodes);
tmp.nodeLabels=zeros(tmp.nodeLabelsDim,tmp.nNodes);
tmp.targets=-ones(1,tmp.nNodes);


c=rand(dataSet.config.subMatDim)>(1-dataSet.config.graphDensity/2);
dataSet.info.targetMatrix=connectMatrix( (c' | c) &(eye(dataSet.config.subMatDim)==0));
dataSet.info.targetLabels=round(dataSet.config.maxLabel*rand(dataSet.config.nodeLabelsDim,dataSet.config.subMatDim));

casSub=0;
 
for it1=1:(tmp.graphNum)
    d=rand(tmp.graphDim)>(1-dataSet.config.graphDensity/2);
    a=connectMatrix((d'|d)&(eye(tmp.graphDim)==0));
    b=round(tmp.maxLabel*rand(tmp.nodeLabelsDim,tmp.graphDim));
    
    indexes=randperm(tmp.graphDim);
    selIndexes=indexes(1:tmp.subMatDim);
    a(selIndexes,selIndexes)=dataSet.info.targetMatrix;
    b(:,selIndexes)=dataSet.info.targetLabels;
    
    [nt,num]=subgraphs(a,b,dataSet.info.targetMatrix,dataSet.info.targetLabels);
    
    if(num>1) 
        casSub=casSub+1;
    end
    nSelIndexes=find(nt);
    
    inIndexes=((it1-1)*tmp.graphDim+1):1:(it1*tmp.graphDim);
    tmp.connMatrix(inIndexes,inIndexes)=a;
    tmp.nodeLabels(:,inIndexes)=b;  
    tmp.targets(:,(it1-1)*tmp.graphDim+nSelIndexes)=1;
    %it1
end

tmp.nodeLabels=tmp.nodeLabels+dataSet.config.normalNoiseCoefficient*randn(size(tmp.nodeLabels))...
    + dataSet.config.uniformNoiseCoefficient*rand(size(tmp.nodeLabels))+ dataSet.config.noiseMean;

dataSet.trainSet.nNodes=dataSet.trainSet.graphNum*dataSet.config.graphDim;
dataSet.testSet.nNodes=dataSet.testSet.graphNum*dataSet.config.graphDim;
dataSet.validationSet.nNodes=dataSet.validationSet.graphNum*dataSet.config.graphDim;

trainIndexes=1:dataSet.trainSet.nNodes;
testIndexes=(dataSet.trainSet.nNodes+1):(dataSet.trainSet.nNodes+dataSet.testSet.nNodes);
validationIndexes=(dataSet.trainSet.nNodes+dataSet.testSet.nNodes+1):(dataSet.trainSet.nNodes+dataSet.testSet.nNodes+dataSet.validationSet.nNodes);

%GRAPH GENERATION END

%In the following the graphs are subdivided into validation, test and
%training set

dataSet.trainSet.maskMatrix=tmp.maskMatrix(trainIndexes,trainIndexes);
dataSet.trainSet.connMatrix=tmp.connMatrix(trainIndexes,trainIndexes);
dataSet.trainSet.nodeLabels=tmp.nodeLabels(:,trainIndexes);
dataSet.trainSet.targets=tmp.targets(:,trainIndexes);

dataSet.testSet.maskMatrix=tmp.maskMatrix(testIndexes,testIndexes);
dataSet.testSet.connMatrix=tmp.connMatrix(testIndexes,testIndexes);
dataSet.testSet.nodeLabels=tmp.nodeLabels(:,testIndexes);
dataSet.testSet.targets=tmp.targets(:,testIndexes);

dataSet.validationSet.maskMatrix=tmp.maskMatrix(validationIndexes,validationIndexes);
dataSet.validationSet.connMatrix=tmp.connMatrix(validationIndexes,validationIndexes);
dataSet.validationSet.nodeLabels=tmp.nodeLabels(:,validationIndexes);
dataSet.validationSet.targets=tmp.targets(:,validationIndexes);



% Check the correctness of the couple <parameter,value>
function check_valueok(name,value,n)
if isempty(value)
    err(n,['makeSubGraphDataset: No value specified for <' name '>']);
end
global params dataSet
switch name
    case 'subMatDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.subMatDim=v;
        params(1)=1;
    case 'graphDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.graphDim=v;
        params(2)=1;
    case 'maxLabel'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.maxLabel=v;
        params(3)=1;
    case 'nodeLabelsDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.nodeLabelsDim=v;
        params(4)=1;
    case 'graphDensity'
        v=str2double(value);
        if isnan(v) || (v<=0) || (v>1)
            err(n,['Parameter <' name '> should be in the interval (0,1]. Check "' value '"']);
        end
        dataSet.config.graphDensity=v;
        params(5)=1;
    case 'normalNoiseCoefficient'
        v=str2double(value);
        if isnan(v) || (v<0)
            err(n,['Parameter <' name '> should be a number greater or equal to zero. Check "' value '"']);
        end
        dataSet.config.normalNoiseCoefficient=v;
        params(6)=1;
    case 'uniformNoiseCoefficient'
        v=str2double(value);
        if isnan(v) || (v<0)
            err(n,['Parameter <' name '> should be a number greater or equal to zero. Check "' value '"']);
        end
        dataSet.config.uniformNoiseCoefficient=v;
        params(7)=1;
    case 'noiseMean'
        v=str2double(value);
        if isnan(v)
            err(n,['Parameter <' name '> should be a number. Check "' value '"']);
        end
        dataSet.config.noiseMean=v;
        params(8)=1;
    case 'rejectUpperThreshold'
        v=str2double(value);
        if isnan(v) || (v<0) || (v>=1)
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        if params(10) && v<dataSet.config.rejectLowerThreshold
            err(n,'Parameters <rejectLowerThreshold> should be less or equal to <rejectUpperThreshold>');
        end
        dataSet.config.rejectUpperThreshold=v;
        params(9)=1;
    case 'rejectLowerThreshold'
        v=str2double(value);
        if isnan(v) || (v<0) || (v>=1)
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        if params(9) && v>dataSet.config.rejectUpperThreshold
            err(n,'Parameters <rejectLowerThreshold> should be less or equal to <rejectUpperThreshold>');
        end
        dataSet.config.rejectLowerThreshold=v;
        params(10)=1;
    case 'ThresholdPass'
        v=str2double(value);
        if isnan(v) 
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        dataSet.config.ThresholdPass=v;
    case 'ThresholdUp'
        v=str2double(value);
        if isnan(v) 
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        dataSet.config.ThresholdUp=v;
    case 'trainSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.trainSet.graphNum=v;
        params(11)=1;
    case 'validationSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.validationSet.graphNum=v;
        params(12)=1;
    
    case 'testSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.testSet.graphNum=v;
        params(13)=1;
    otherwise
        err(n,['Parameter <' name '> is unknown']);
end


% Check if all necessary parameters has been set
function check_thereisall
global params params_name
ok=1;
for i=1:size(params,2)
    if params(i)==0
        try
            err(0,['makeSubGraphDataset: No parameter <' char(params_name(i)) '> in the configuration file']);
        catch
            ok=0;
        end
    end
end
if ok==0
    rethrow(lasterr);
end