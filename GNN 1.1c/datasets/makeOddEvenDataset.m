% The function build a dataset composed by random graphs.
% The targets of each node is:
%       +1 if its label (that is an integer) is even
%       -1 if its label is odd
function makeOddEvenDataset(file)
% Create a dataset for the odd-even recognition problem
% USAGE: makeOddEvenDataset(<file.config>)

global dataSet params params_name
if ~isempty(dataSet)
    dataSet=[];
end


% load parameters from file
try
    if nargin ~= 1
        file='OddEvenDataset.config';     %default config file
    end
    [fid,message] = fopen(file,'rt');
    if fid == -1
        err(0,['<' file '>: ' message]);
    end
    
    
    params = [0 0 0 0 0 0 0 0];
    % graphDim maxLabel graphDensity rejectUpperThreshold rejectLowerThreshold 
    % trainSet.graphNum validationSet.graphNum testSet.graphNum
    params_name={'graphDim' 'maxLabel' 'graphDensity' 'rejectUpperThreshold' 'rejectLowerThreshold'...
                 'trainSet.graphNum' 'validationSet.graphNum' 'testSet.graphNum'};
    
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



%GRAPH GENERATION START
dataSet.config.type='classification';
dataSet.config.nodeLabelsDim=1;


tmp.graphNum=dataSet.trainSet.graphNum+dataSet.testSet.graphNum+dataSet.validationSet.graphNum;
tmp.graphDim=dataSet.config.graphDim;
tmp.nNodes=tmp.graphNum*tmp.graphDim;
tmp.maskMatrix=speye(tmp.nNodes);


tmp.nodeLabelsDim=dataSet.config.nodeLabelsDim;
tmp.maxLabel=dataSet.config.maxLabel;
tmp.connMatrix=sparse(tmp.nNodes,tmp.nNodes);

tmp.nodeLabels=zeros(tmp.nodeLabelsDim,tmp.nNodes);

tmp.targets=-ones(1,tmp.nNodes);


addpath(genpath(pwd));

for it1=1:(tmp.graphNum)
    d=rand(tmp.graphDim)>(1-dataSet.config.graphDensity/2);
    a=connectMatrix((d'|d)&(eye(tmp.graphDim)==0));
    
    
    
    b=round(tmp.maxLabel*rand(tmp.nodeLabelsDim,tmp.graphDim));
                
    inIndexes=((it1-1)*tmp.graphDim+1):1:(it1*tmp.graphDim);
    tmp.connMatrix(inIndexes,inIndexes)=a;
    
    tmp.nodeLabels(:,inIndexes)=b;  
end
for i=1:tmp.nNodes
    if mod(tmp.nodeLabels(i),2)==0
        tmp.targets(i)=1;
    end
end



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
    err(n,['MakeOddEvenDataset: No value specified for <' name '>']);
end
global params dataSet
switch name
    case 'graphDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.graphDim=v;
        params(1)=1;
    case 'maxLabel'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.maxLabel=v;
        params(2)=1;
    case 'graphDensity'
        v=str2double(value);
        if isnan(v) || (v<=0) || (v>=1)
            err(n,['Parameter <' name '> should be in the interval (0,1). Check "' value '"']);
        end
        dataSet.config.graphDensity=v;
        params(3)=1;
    case 'rejectUpperThreshold'
        v=str2double(value);
        if isnan(v) || (v<0) || (v>=1)
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        if params(5) && v<dataSet.config.rejectLowerThreshold
            err(n,'Parameters <rejectLowerThreshold> should be less or equal to <rejectUpperThreshold>');
        end
        dataSet.config.rejectUpperThreshold=v;
        params(4)=1;
    case 'rejectLowerThreshold'
        v=str2double(value);
        if isnan(v) || (v<0) || (v>=1)
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        if params(4) && v>dataSet.config.rejectUpperThreshold
            err(n,'Parameters <rejectLowerThreshold> should be less or equal to <rejectUpperThreshold>');
        end
        dataSet.config.rejectLowerThreshold=v;
        params(5)=1;
    case 'trainSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.trainSet.graphNum=v;
        params(6)=1;
    case 'validationSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.validationSet.graphNum=v;
        params(7)=1;
    case 'testSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.testSet.graphNum=v;
        params(8)=1;
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
            err(0,['makeOddEvenDataset: No parameter <' char(params_name(i)) '> in the configuration file']);
        catch
            ok=0;
        end
    end
end
if ok==0
    rethrow(lasterr);
end