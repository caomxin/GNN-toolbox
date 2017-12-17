function makeTreeDepthDataset(file)
% Create a dataset for the TreeDepth problem
% USAGE: makeTreeDepthDataset(<file.config>)


global dataSet params params_name
if ~isempty(dataSet)
    dataSet=[];
end


% load parameters from file
try
    if nargin ~= 1
        file='TreeDepthDataset.config';     %default config file
    end
    [fid,message] = fopen(file,'rt');
    if fid == -1
        err(0,['<' file '>: ' message]);
    end
    
    params = [0 0 0 0 0 0];
    % treeDim nodeLabelsDim maxLabel trainSet.supervisedNodeNumber
    % validationSet.supervisedNodeNumber testSet.supervisedNodeNumber
    params_name={'treeDim' 'nodeLabelsDim' 'maxLabel'...
        'trainSet.supervisedNodeNumber' 'validationSet.supervisedNodeNumber' 'testSet.supervisedNodeNumber'};
    
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


dataSet.config.type='regression';

%GRAPH GENERATION START

dataSet.config.supervisedNodeNumber=dataSet.trainSet.supervisedNodeNumber+dataSet.testSet.supervisedNodeNumber+...
    dataSet.validationSet.supervisedNodeNumber;

dataSet.trainSet.nNodes=dataSet.config.graphDim;
dataSet.testSet.nNodes=dataSet.config.graphDim;
dataSet.validationSet.nNodes=dataSet.config.graphDim;


b=randperm(dataSet.config.supervisedNodeNumber);

dataSet.trainSet.supervisedNodes=b(1:dataSet.trainSet.supervisedNodeNumber);
dataSet.validationSet.supervisedNodes=b((dataSet.trainSet.supervisedNodeNumber+1):...
    (dataSet.trainSet.supervisedNodeNumber+dataSet.validationSet.supervisedNodeNumber));
dataSet.testSet.supervisedNodes=b((dataSet.trainSet.supervisedNodeNumber+dataSet.validationSet.supervisedNodeNumber+1):...
    (dataSet.config.supervisedNodeNumber));
dataSet.config.supervisedNodes=b;


dataSet.config.maxChildren=round(log(dataSet.config.graphDim)/2);
if dataSet.config.maxChildren<2;
    dataSet.config.maxChildren=2;
end

nodes=randperm(dataSet.config.graphDim);
curInd=1;
done=false;
fathers=nodes(1);
targets(fathers)=0;
newfathers=[];
level=0;


dataSet.config.connMatrix=sparse([],[],[],dataSet.config.graphDim,dataSet.config.graphDim,dataSet.config.graphDim);
dataSet.config.nodeLabels=round(dataSet.config.maxLabel*rand(dataSet.config.nodeLabelsDim,dataSet.config.graphDim));
dataSet.config.targets=zeros(1,dataSet.config.graphDim);

while done==false
    level=level+1;
    %disp(['******************* level ' num2str(level) ' ************************']);
    for i=1:size(fathers,2)
        %disp(['i = ' num2str(i)]);
        currentnode=fathers(i);
        nCh=round(dataSet.config.maxChildren*rand)+1;
        if curInd+nCh < dataSet.config.graphDim
            children=nodes(curInd+1:curInd+nCh);
            newfathers=[newfathers children];
        else
            children=nodes(curInd+1:end);
            done=true;
        end
        %chain current node with its children
        dataSet.config.connMatrix(currentnode,children)=1;
        dataSet.config.targets(children)=level;
        if done 
            break;
        end
        curInd=curInd+nCh;
    end
    fathers=newfathers;
    newfathers=[];
end


% %scaling of the targets in the interval 0:1
% mintarget=min(dataSet.config.targets)-1;
% maxtarget=max(dataSet.config.targets);
% dataSet.config.targets=((dataSet.config.targets-mintarget)/(maxtarget-mintarget))*1;





%In the following the validation, the test and the
%training set is built from the graph in dataset
dataSet.trainSet.connMatrix=dataSet.config.connMatrix;
dataSet.testSet.connMatrix=dataSet.config.connMatrix;
dataSet.validationSet.connMatrix=dataSet.config.connMatrix;

dataSet.trainSet.nodeLabels=dataSet.config.nodeLabels;
dataSet.testSet.nodeLabels=dataSet.config.nodeLabels;
dataSet.validationSet.nodeLabels=dataSet.config.nodeLabels;

dataSet.trainSet.targets=dataSet.config.targets;
dataSet.testSet.targets=dataSet.config.targets;
dataSet.validationSet.targets=dataSet.config.targets;

dataSet.trainSet.maskMatrix=spdiags(sparse(dataSet.trainSet.supervisedNodes,...
    ones(dataSet.trainSet.supervisedNodeNumber,1),ones(dataSet.trainSet.supervisedNodeNumber,1), ...
    dataSet.config.graphDim,1),0,dataSet.config.graphDim,dataSet.config.graphDim);
dataSet.testSet.maskMatrix=spdiags(sparse(dataSet.testSet.supervisedNodes,...
    ones(dataSet.testSet.supervisedNodeNumber,1),ones(dataSet.testSet.supervisedNodeNumber,1), ...
    dataSet.config.graphDim,1),0,dataSet.config.graphDim,dataSet.config.graphDim);
dataSet.validationSet.maskMatrix=spdiags(sparse(dataSet.validationSet.supervisedNodes,...
    ones(dataSet.validationSet.supervisedNodeNumber,1),ones(dataSet.validationSet.supervisedNodeNumber,1),...
    dataSet.config.graphDim,1),0,dataSet.config.graphDim,dataSet.config.graphDim);



% Check the correctness of the couple <parameter,value>
function check_valueok(name,value,n)
if isempty(value)
    err(n,['MakeGrandParentsDataset: No value specified for <' name '>']);
end
global params dataSet
switch name
    case 'treeDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        if params(4) && params(5) && params(6) && ...
             v<(dataSet.trainSet.supervisedNodeNumber+dataSet.validationSet.supervisedNodeNumber+dataSet.testSet.supervisedNodeNumber)
            err(n,'Parameters <graphDim> should be greater or equal to (<trainSet.supervisedNodeNumber>+<validationSet.supervisedNodeNumber>+<testSet.supervisedNodeNumber>');
        end
        dataSet.config.graphDim=v;
        params(1)=1;
    case 'nodeLabelsDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.nodeLabelsDim=v;
        params(2)=1;
    case 'maxLabel'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.maxLabel=v;
        params(3)=1;
    case 'trainSet.supervisedNodeNumber'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        if params(1) && params(5) && params(6) && ...
             v>(dataSet.config.graphDim-dataSet.validationSet.supervisedNodeNumber-dataSet.testSet.supervisedNodeNumber)
            err(n,'Parameters <graphDim> should be greater or equal to (<trainSet.supervisedNodeNumber>+<validationSet.supervisedNodeNumber>+<testSet.supervisedNodeNumber>');
        end
        dataSet.trainSet.supervisedNodeNumber=v;
        params(4)=1;
    case 'validationSet.supervisedNodeNumber'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        if params(1) && params(4) && params(6) && ...
             v>(dataSet.config.graphDim-dataSet.trainSet.supervisedNodeNumber-dataSet.testSet.supervisedNodeNumber)
            err(n,'Parameters <graphDim> should be greater or equal to (<trainSet.supervisedNodeNumber>+<validationSet.supervisedNodeNumber>+<testSet.supervisedNodeNumber>');
        end
        dataSet.validationSet.supervisedNodeNumber=v;
        params(5)=1;
    case 'testSet.supervisedNodeNumber'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        if params(1) && params(4) && params(5) && ...
             v>(dataSet.config.graphDim-dataSet.trainSet.supervisedNodeNumber-dataSet.validationSet.supervisedNodeNumber)
            err(n,'Parameters <graphDim> should be greater or equal to (<trainSet.supervisedNodeNumber>+<validationSet.supervisedNodeNumber>+<testSet.supervisedNodeNumber>');
        end
        dataSet.testSet.supervisedNodeNumber=v;
        params(6)=1;
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
            err(0,['MakeGrandParentsDataset: No parameter <' char(params_name(i)) '> in the configuration file']);
        catch
            ok=0;
        end
    end
end
if ok==0
    rethrow(lasterr);
end

