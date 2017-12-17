function makeCliqueDataset_fix_rand(file)
% Create two datasets for the clique recognition problem.
% dataset0 and dataset1 have the same graphs but the first with fixed
% labels, the second with random labels
% USAGE: makeCliqueDataset(<file.config>)


global dataSet0 dataSet1 params params_name randomlabel
if ~isempty(dataSet0)
    dataSet0=[];
end
if ~isempty(dataSet1)
    dataSet1=[];
end



% load parameters from file
try
    if nargin ~= 1
        file='CliqueDataset.config';     %default config file
    end
    [fid,message] = fopen(file,'rt');
    if fid == -1
        err(0,['<' file '>: ' message]);
    end
    
    params = [0 0 0 0 0 0 0 0 0 0];
    % cliqueDim graphDim maxLabel nodeLabelsDim typeLabel graphDensity 
    % rejectUpperThreshold rejectLowerThreshold trainSet.graphNum
    % validationSet.graphNum testSet.graphNum 
    params_name={'cliqueDim' 'graphDim' 'maxLabel' 'nodeLabelsDim' 'typeLabel' 'graphDensity' 'rejectUpperThreshold' 'rejectLowerThreshold' ...
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

dataSet0.config.type='classification';



%GRAPH GENERATION START
tmp.graphNum=dataSet0.trainSet.graphNum+dataSet0.testSet.graphNum+dataSet0.validationSet.graphNum;
tmp.graphDim=dataSet0.config.graphDim;
tmp.nNodes=tmp.graphNum*tmp.graphDim;
tmp.maskMatrix=speye(tmp.nNodes);
tmp.cliqueDim=dataSet0.config.cliqueDim;


tmp.nodeLabelsDim=dataSet0.config.nodeLabelsDim;
tmp.maxLabel=dataSet0.config.maxLabel;
tmp.connMatrix=sparse(tmp.nNodes,tmp.nNodes);

% for random labels
tmp.nodeLabels=zeros(tmp.nodeLabelsDim,tmp.nNodes);

    
tmp.targets=-ones(1,tmp.nNodes);


addpath(genpath(pwd));

for it1=1:(tmp.graphNum)
    d=rand(tmp.graphDim)>(1-dataSet0.config.graphDensity/2);
    a=connectMatrix((d'|d)&(eye(tmp.graphDim)==0));
    
    b=round(tmp.maxLabel*rand(tmp.nodeLabelsDim,tmp.graphDim));

    
    indexes=randperm(tmp.graphDim);
    selIndexes=indexes(1:tmp.cliqueDim);
    a(selIndexes,selIndexes)=1;
    
    %delete selfloops
    for i=1:tmp.graphDim
        a(i,i)=0;
    end
    
    [ind,num]=cliques(a,tmp.cliqueDim);
    foundIndexes=find(ind);
    
    inIndexes=((it1-1)*tmp.graphDim+1):1:(it1*tmp.graphDim);
    tmp.connMatrix(inIndexes,inIndexes)=a;
    
    tmp.nodeLabels(:,inIndexes)=b;  
    
    tmp.targets(:,(it1-1)*tmp.graphDim+foundIndexes)=1;
end


dataSet0.trainSet.nNodes=dataSet0.trainSet.graphNum*dataSet0.config.graphDim;
dataSet0.testSet.nNodes=dataSet0.testSet.graphNum*dataSet0.config.graphDim;
dataSet0.validationSet.nNodes=dataSet0.validationSet.graphNum*dataSet0.config.graphDim;

trainIndexes=1:dataSet0.trainSet.nNodes;
testIndexes=(dataSet0.trainSet.nNodes+1):(dataSet0.trainSet.nNodes+dataSet0.testSet.nNodes);
validationIndexes=(dataSet0.trainSet.nNodes+dataSet0.testSet.nNodes+1):(dataSet0.trainSet.nNodes+dataSet0.testSet.nNodes+dataSet0.validationSet.nNodes);

%GRAPH GENERATION END

%In the following the graphs are subdivided into validation, test and
%training set

dataSet0.trainSet.maskMatrix=tmp.maskMatrix(trainIndexes,trainIndexes);
dataSet0.trainSet.connMatrix=tmp.connMatrix(trainIndexes,trainIndexes);
dataSet0.trainSet.nodeLabels=tmp.nodeLabels(:,trainIndexes);
dataSet0.trainSet.targets=tmp.targets(:,trainIndexes);

dataSet0.testSet.maskMatrix=tmp.maskMatrix(testIndexes,testIndexes);
dataSet0.testSet.connMatrix=tmp.connMatrix(testIndexes,testIndexes);
dataSet0.testSet.nodeLabels=tmp.nodeLabels(:,testIndexes);
dataSet0.testSet.targets=tmp.targets(:,testIndexes);

dataSet0.validationSet.maskMatrix=tmp.maskMatrix(validationIndexes,validationIndexes);
dataSet0.validationSet.connMatrix=tmp.connMatrix(validationIndexes,validationIndexes);
dataSet0.validationSet.nodeLabels=tmp.nodeLabels(:,validationIndexes);
dataSet0.validationSet.targets=tmp.targets(:,validationIndexes);

% fixed labels
dataSet1=dataSet0;
dataSet1.trainSet.nodeLabels = ones(size(dataSet1.trainSet.nodeLabels));
dataSet1.validationSet.nodeLabels = ones(size(dataSet1.validationSet.nodeLabels));
dataSet1.testSet.nodeLabels = ones(size(dataSet1.testSet.nodeLabels));




% -------------- Auxiliary functions --------------- %
% Check the correctness of the couple <parameter,value>
function check_valueok(name,value,n)
if isempty(value)
    err(n,['MakeCliqueDataset: No value specified for <' name '>']);
end
global params dataSet0 randomlabel
switch name
    case 'cliqueDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        if params(2) && v>dataSet0.config.graphDim
            err(n,'Parameters <cliqueDim> should be less or equal to <graphDim>');
        end
        dataSet0.config.cliqueDim=v;
        params(1)=1;
    case 'graphDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        if params(1) && v<dataSet0.config.cliqueDim
            err(n,'Parameters <graphDim> should be greater or equal to <cliqueDim>');
        end
        dataSet0.config.graphDim=v;
        params(2)=1;
    case 'maxLabel'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet0.config.maxLabel=v;
        params(3)=1;
    case 'nodeLabelsDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet0.config.nodeLabelsDim=v;
        params(4)=1;
    case 'typeLabel'
        if strcmp(value,'random')
            randomlabel=1;
        else
            randomlabel=0;
        end
        params(5)=1;
    case 'graphDensity'
        v=str2double(value);
        if isnan(v) || (v<=0) || (v>1)
            err(n,['Parameter <' name '> should be in the interval (0,1]. Check "' value '"']);
        end
        dataSet0.config.graphDensity=v;
        params(6)=1;
    case 'rejectUpperThreshold'
        v=str2double(value);
        if isnan(v) || (v<0) || (v>=1)
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        if params(8) && v<dataSet0.config.rejectLowerThreshold
            err(n,'Parameters <rejectLowerThreshold> should be less or equal to <rejectUpperThreshold>');
        end
        dataSet0.config.rejectUpperThreshold=v;
        params(7)=1;
    case 'rejectLowerThreshold'
        v=str2double(value);
        if isnan(v) || (v<0) || (v>=1)
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        if params(7) && v>dataSet0.config.rejectUpperThreshold
            err(n,'Parameters <rejectLowerThreshold> should be less or equal to <rejectUpperThreshold>');
        end
        dataSet0.config.rejectLowerThreshold=v;
        params(8)=1;
    case 'trainSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet0.trainSet.graphNum=v;
        params(9)=1;
    case 'validationSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet0.validationSet.graphNum=v;
        params(10)=1;
    case 'testSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet0.testSet.graphNum=v;
        params(11)=1;
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
            err(0,['MakeCliqueDataset: No parameter <' char(params_name(i)) '> in the configuration file']);
        catch
            ok=0;
        end
    end
end
if ok==0
    rethrow(lasterr);
end