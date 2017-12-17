% The function builds a graph with labels (l1,l2), where l1 and l2 are binary values. 
% The goal is to attatch to each node an output which should be
%   - the node pagerank if the node label is (0,0) or (1,1)
%   - the node pagerank times 2, otherwise.
%
% The function
%   1. builds  a random graph:
%   2. attatches to each  node a random label (l1,l2), where l1 and l2 are binaries
%   3. The pageRank x_n is computed for each node of the graph
%   4. A a target is attatched to each node
%           - x_n if  the node label is (0,0) or (1,1)
%           - 2* x_n otherwise
%   5. it is randomly decided which nodes must be supervised and which must not
%   6. a train, a validation and a test set are build: the graph of the three datasets 
%       is the same, but the supervised nodes are different


% REMARK: MODIFYING THIS PROCEDURE
% In order to replace the random graphs with graphs acquired from any dataset,
% the section "GRAPH GENERATION" must be appropriately replaced
%
% tmp.connMatrix 
% It is the transposed connection matrix N x N: element in position (i,j) is 1
% if there is an arc from i to j and 0 otherwise; N is graphDim*graphNum
% tmp.connMatrix represent all the graphs in the  dataset
% (train+validation+test)
%
% tmp.nodeLabels
% It is a LxN matrix containing the labels of the graph: L is the dimension of
% each label
% (currently L=2, but the procudere works for any L)
%
% tmp.targets 
% It is a = O x N matrix containing the desired output for each node: O is
% the output dimension
% (currently O=1, but the procudere works for any O)
%
% tmp.maskMatrix 
% It is diagonal N x N matrix where the element d_i in position i is 1 if the
% the i-th element of the output must be supervised and 0 oteherwise.
% In other words, the error function is (t-o)' maskMatrix (t-o), where
% t is the target and o the output

function makeWebPagesScoringDataset(file)
% Create a dataset for the Ranking problem
% USAGE: makeWebPagesScoringDataset(<file.config>)

global dataSet params params_name
if ~isempty(dataSet)
    dataSet=[];
end


% load parameters from file
try
    if nargin ~= 1
        file='WebPagesScoringDataset.config';     %default config file
    end
    [fid,message] = fopen(file,'rt');
    if fid == -1
        err(0,['<' file '>: ' message]);
    end
    

    params = [0 0 0 0 0 0 0];
    % maxLabel graphDim nodeLabelsDim graphDensity trainSet.supervisedNodeNumber 
    % validationSet.supervisedNodeNumber testSet.supervisedNodeNumber
    params_name={'maxLabel' 'graphDim' 'nodeLabelsDim' 'graphDensity' ...
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

c=(rand(dataSet.config.graphDim)>(1-dataSet.config.graphDensity/2)&(eye(dataSet.config.graphDim)==0));
dataSet.config.connMatrix=sparse(c);
dataSet.config.nodeLabels=round(dataSet.config.maxLabel*rand(dataSet.config.nodeLabelsDim,dataSet.config.graphDim));

[r,c]=find(dataSet.config.connMatrix);
s=1 ./ sum(dataSet.config.connMatrix);
W=sparse(r,c,s(c)*0.85);
E=sparse(ones(dataSet.config.graphDim,1));
X=sparse([],[],[],dataSet.config.graphDim,1);
n=100;
X=getPR(X,W,E,n);

dataSet.config.importantNodes=find(sum(dataSet.config.nodeLabels(:,:))==1);
dataSet.config.nonImportantNodes=setdiff(1:dataSet.config.graphDim,dataSet.config.importantNodes);
dataSet.config.targets=sparse(1,dataSet.config.graphDim);
dataSet.config.targets(dataSet.config.importantNodes)=2*X(dataSet.config.importantNodes)';
dataSet.config.targets(dataSet.config.nonImportantNodes)=X(dataSet.config.nonImportantNodes)';

%GRAPH GENERATION END

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
    err(n,['MakeGraphDataset: No value specified for <' name '>']);
end
global params dataSet
switch name
    case 'maxLabel'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.maxLabel=v;
        params(1)=1;
    case 'graphDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.graphDim=v;
        params(2)=1;
    case 'nodeLabelsDim'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        dataSet.config.nodeLabelsDim=v;
        params(3)=1;
    case 'graphDensity'
        v=str2double(value);
        if isnan(v) || (v<=0) || (v>1)
            err(n,['Parameter <' name '> should be in the interval (0,1]. Check "' value '"']);
        end
        dataSet.config.graphDensity=v;
        params(4)=1;
    case 'trainSet.supervisedNodeNumber'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        if params(2) && params(6) && params(7) && ...
                v+dataSet.validationSet.supervisedNodeNumber+dataSet.testSet.supervisedNodeNumber>dataSet.config.graphDim
            err(n,'Total number of supervised node should be less or equal to <graphDim>');
        end
        dataSet.trainSet.supervisedNodeNumber=v;
        params(5)=1;
    case 'validationSet.supervisedNodeNumber'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        if params(2) && params(5) && params(7) && ...
                v+dataSet.trainSet.supervisedNodeNumber+dataSet.testSet.supervisedNodeNumber>dataSet.config.graphDim
            err(n,'Total number of supervised node should be less or equal to <graphDim>');
        end
        dataSet.validationSet.supervisedNodeNumber=v;
        params(6)=1;
    case 'testSet.supervisedNodeNumber'
        v=str2double(value);
        if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
            err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);
        end
        if params(2) && params(5) && params(6) && ...
                v+dataSet.trainSet.supervisedNodeNumber+dataSet.validationSet.supervisedNodeNumber>dataSet.config.graphDim
            err(n,'Total number of supervised node should be less or equal to <graphDim>');
        end
        dataSet.testSet.supervisedNodeNumber=v;
        params(7)=1;
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
            err(0,['makeWebPagesScoringDataset: No parameter <' char(params_name(i)) '> in the configuration file']);
        catch
            ok=0;
        end
    end
end
if ok==0
    rethrow(lasterr);
end


