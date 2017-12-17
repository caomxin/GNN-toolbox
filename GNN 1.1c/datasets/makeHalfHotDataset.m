function makeHalfHotDataset
% Create a dataset for the Half Hot problem
% USAGE: makeHalfHotDataset(<file.config>)

global dataSet params params_name
if ~isempty(dataSet)
    dataSet=[];
end


% load parameters from file
try
    if nargin ~= 1
        file='HalfHotDataset.config';     %default config file
    end
    [fid,message] = fopen(file,'rt');
    if fid == -1
        err(0,['<' file '>: ' message]);
    end
    
    params = [0 0 0 0 0 0 0 0 0 0];
    % maxGraphDim maxOrder attemptNum nodeLabelsDim maxLabel 
    % rejectUpperThreshold rejectLowerThreshold trainSet.graphNum validationSet.graphNum testSet.graphNum
    params_name={'maxGraphDim' 'maxOrder' 'attemptNum' 'nodeLabelsDim' 'maxLabel' ...
        'rejectUpperThreshold' 'rejectLowerThreshold' 'trainSet.graphNum'...
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


dataSet.config.maxGraph=0;
dataSet.config.graphNum=dataSet.trainSet.graphNum+dataSet.validationSet.graphNum+dataSet.testSet.graphNum;

dataSet.config.type='classification';

for i=1:dataSet.config.graphNum,
    notDone=1;
    while notDone
        notDone2=1;
        while notDone2,
            sDim=dataSet.config.subGraphDimRange(floor(1+rand*length(dataSet.config.subGraphDimRange)));  % graphDimRange (error)
            gDim=dataSet.config.graphDimRange(floor(1+rand*length(dataSet.config.graphDimRange)));
            if gDim>sDim && (mod(gDim,2)==0 || mod(sDim,2)==0),
                notDone2=0;
            end
        end
        f=0;
        for j=1:dataSet.config.attemptNum,
            [g,f]=generateUniformGraph(gDim,sDim);
            if f==0,
                break;
            end
        end
        if (f==1)
            fprintf('Graph %d: Failed for gDim=%d, sDim=%d\n',i,gDim,sDim);
        else
            notDone=0;
            ds(i).connMatrix=g;
            ds(i).dim=gDim;
            ds(i).subDim=sDim;
            ds(i).maskMatrix=ones(gDim)-eye(gDim);
            ds(i).nodeLabels=floor(rand(dataSet.config.nodeLabelsDim,gDim)*(dataSet.config.maxLabel+1));
            ds(i).targets=sparse([],[],[],gDim,1);
            
            % GABRI 27-02-2006 
            %devo aver perso l'assegnazione originale dei targets fatta da Franco. 
            %Mi pare che così si assegni a caso target positivo a metà dei nodi di ogni grafo
            index_tg=randperm(gDim);
            ds(i).targets(index_tg(1:gDim/2))=1;
            ds(i).targets(index_tg(gDim/2+1:gDim))=-1;
            
            expDis=factorial(gDim) ./(factorial(gDim-(0:gDim)) .*factorial((0:gDim)))';
            ds(i).expectedDistribution=expDis / sum(expDis);
            dataSet.config.maxGraph=max(gDim,dataSet.config.maxGraph);
            %fprintf('Graph %d: Done for gDim=%d, sDim=%d at %d\n',i,gDim,sDim,j);
        end
        
    end
end
g1=1:length(ds);
t1=randperm(length(ds));
dataSet.config.trainGraphs=t1(1:dataSet.trainSet.graphNum);
g2=setdiff(g1,dataSet.config.trainGraphs);
t2=randperm(length(g2));
dataSet.config.validationGraphs=g2(t2(1:dataSet.validationSet.graphNum));
g3=setdiff(g2,dataSet.config.validationGraphs);
t3=randperm(length(g3));
dataSet.config.testGraphs=g3(t3(1:dataSet.testSet.graphNum));

c=1;
dataSet.trainSet.maskMatrix=sparse([],[],[]);
dataSet.trainSet.connMatrix=sparse([],[],[]);
dataSet.trainSet.nodeLabels=sparse([],[],[]);
dataSet.trainSet.expectedDistribution=zeros(dataSet.config.maxGraph+1,1);
dataSet.trainSet.targets=zeros(1,4*dataSet.trainSet.graphNum); % is the minimum length
dataSet.trainSet.graphRanges=zeros(dataSet.trainSet.graphNum,2);
for i=1:dataSet.trainSet.graphNum,
%     i
    ind=dataSet.config.trainGraphs(i);
    d=ds(ind).dim;
    dataSet.trainSet.connMatrix(c:(c+d-1),c:(c+d-1))=ds(ind).connMatrix;
    dataSet.trainSet.nodeLabels(:,c:(c+d-1))=ds(ind).nodeLabels;
    dataSet.trainSet.maskMatrix(c:(c+d-1),c:(c+d-1))=ds(ind).maskMatrix;
    dataSet.trainSet.targets(c:(c+d-1))=ds(ind).targets;
    dataSet.trainSet.graphRanges(i,:)=[c,c+d-1];
    il=((dataSet.config.maxGraph-d)/2+1):((dataSet.config.maxGraph+d)/2+1);
%     ds(ind).expectedDistribution   %%%%
    dataSet.trainSet.expectedDistribution(il)=dataSet.trainSet.expectedDistribution(il)+ds(ind).expectedDistribution;
    c=c+d;
end
dataSet.trainSet.nNodes=length(dataSet.trainSet.targets);
dataSet.trainSet.expectedDistribution=dataSet.trainSet.expectedDistribution/dataSet.trainSet.graphNum;

c=1;
dataSet.validationSet.maskMatrix=sparse([],[],[]);
dataSet.validationSet.connMatrix=sparse([],[],[]);
dataSet.validationSet.nodeLabels=sparse([],[],[]);
dataSet.validationSet.expectedDistribution=zeros(dataSet.config.maxGraph+1,1);
dataSet.validationSet.targets=zeros(1,4*dataSet.validationSet.graphNum); % is the minimum length
dataSet.validationSet.graphRanges=zeros(dataSet.validationSet.graphNum,2);
for i=1:dataSet.validationSet.graphNum,
    ind=dataSet.config.validationGraphs(i);
    d=ds(ind).dim;
    dataSet.validationSet.connMatrix(c:(c+d-1),c:(c+d-1))=ds(ind).connMatrix;
    dataSet.validationSet.nodeLabels(:,c:(c+d-1))=ds(ind).nodeLabels;
    dataSet.validationSet.maskMatrix(c:(c+d-1),c:(c+d-1))=ds(ind).maskMatrix;
    dataSet.validationSet.targets(c:(c+d-1))=ds(ind).targets;
    dataSet.validationSet.graphRanges(i,:)=[c,c+d-1];
    il=((dataSet.config.maxGraph-d)/2+1):((dataSet.config.maxGraph+d)/2+1);
    dataSet.validationSet.expectedDistribution(il)=dataSet.validationSet.expectedDistribution(il)+ds(ind).expectedDistribution;
    c=c+d;
end
dataSet.validationSet.nNodes=length(dataSet.validationSet.targets);
dataSet.validationSet.expectedDistribution=dataSet.validationSet.expectedDistribution/dataSet.validationSet.graphNum;

c=1;
dataSet.testSet.maskMatrix=sparse([],[],[]);
dataSet.testSet.connMatrix=sparse([],[],[]);
dataSet.testSet.nodeLabels=sparse([],[],[]);
dataSet.testSet.expectedDistribution=zeros(dataSet.config.maxGraph+1,1);
dataSet.testSet.targets=zeros(1,4*dataSet.testSet.graphNum); % is the minimum length
dataSet.testSet.graphRanges=zeros(dataSet.testSet.graphNum,2);
for i=1:dataSet.testSet.graphNum,
    ind=dataSet.config.testGraphs(i);
    d=ds(ind).dim;
    dataSet.testSet.connMatrix(c:(c+d-1),c:(c+d-1))=ds(ind).connMatrix;
    dataSet.testSet.nodeLabels(:,c:(c+d-1))=ds(ind).nodeLabels;
    dataSet.testSet.maskMatrix(c:(c+d-1),c:(c+d-1))=ds(ind).maskMatrix;
    dataSet.testSet.targets(c:c+d-1)=ds(ind).targets;
    dataSet.testSet.graphRanges(i,:)=[c,c+d-1];
    il=((dataSet.config.maxGraph-d)/2+1):((dataSet.config.maxGraph+d)/2+1);
    dataSet.testSet.expectedDistribution(il)=dataSet.testSet.expectedDistribution(il)+ds(ind).expectedDistribution;
  c=c+d;
end
dataSet.testSet.nNodes=length(dataSet.testSet.targets);
dataSet.testSet.expectedDistribution=dataSet.testSet.expectedDistribution/dataSet.testSet.graphNum;


% Check the correctness of the couple <parameter,value>
function check_valueok(name,value,n)
if isempty(value)
    err(n,['makeHalfHotDataset: No value specified for <' name '>']);
end
global params dataSet
switch name
    case 'maxGraphDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=4)
            err(n,['Parameter <' name '> should be a positive integer greater than 4. Check "' value '"']);
        end
        dataSet.config.graphDimRange=4:2:v;
        params(1)=1;
    case 'maxOrder'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=3)
            err(n,['Parameter <' name '> should be a positive integer greater than 3. Check "' value '"']);
        end
        dataSet.config.subGraphDimRange=3:v;
        params(2)=1;
    case 'attemptNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.attemptNum=v;
        params(3)=1;
    case 'nodeLabelsDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.nodeLabelsDim=v;
        params(4)=1;
    case 'maxLabel'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.maxLabel=v;
        params(5)=1;
    case 'rejectUpperThreshold'
        v=str2double(value);
        if isnan(v) || (v<0) || (v>=1)
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        if params(7) && v<dataSet.config.rejectLowerThreshold
            err(n,'Parameters <rejectLowerThreshold> should be less or equal to <rejectUpperThreshold>');
        end
        dataSet.config.rejectUpperThreshold=v;
        params(6)=1;
    case 'rejectLowerThreshold'
        v=str2double(value);
        if isnan(v) || (v<0) || (v>=1)
            err(n,['Parameter <' name '> should be in the interval [0,1). Check "' value '"']);
        end
        if params(6) && v>dataSet.config.rejectUpperThreshold
            err(n,'Parameters <rejectLowerThreshold> should be less or equal to <rejectUpperThreshold>');
        end
        dataSet.config.rejectLowerThreshold=v;
        params(7)=1;
    case 'trainSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.trainSet.graphNum=v;
        params(8)=1;
    case 'validationSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.validationSet.graphNum=v;
        params(9)=1;
    
    case 'testSet.graphNum'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.testSet.graphNum=v;
        params(10)=1;
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
            err(0,['makeHalfHotDataset: No parameter <' char(params_name(i)) '> in the configuration file']);
        catch
            ok=0;
        end
    end
end
if ok==0
    rethrow(lasterr);
end
