% function [r,state,i]=linearModelRunForward(maxIt,x,dataset,p,sys,stopCoef)
function [r,state,i]=linearModelRunForward(maxIt,x,dataset,optimalParam)
%% This function just call forward a number "n" of times and compute the new state.

global dynamicSystem dataSet learning

% [f,state.transitionNet]=feval(sys.transitionNet.forwardFunction,dataset.nodeLabels,p.transitionNet);
% sf=size(f);
state.transitionNetState=feval(dynamicSystem.config.transitionNet.forwardFunction,dataSet.(dataset).nodeLabels,'transitionNet',optimalParam);
sf=size(state.transitionNetState.outs);
sqrt_val=sqrt(sf(1));

nLinks=sum(dataSet.(dataset).connMatrix);
outDegreeFactor=1 ./ (nLinks+(nLinks==0));
%f=kron(outDegreeFactor,ones(sf(1),1)) .* f /sqrt(sf(1)) ;


%%%%%%%%%%%%%%% OPTIMIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B = kron(A, ones(m,n)) -->  i = 1:p; i = i(ones(1,m),:);
%                             j = 1:q; j = j(ones(1,n),:);
%                             B = A(i,j);

%transitionNetState.outs=kron(outDegreeFactor,ones(sf(1),1)) .* state.transitionNetState.outs /sqrt(sf(1));
i=ones(sf(1),1); j=1:size(dataSet.(dataset).connMatrix,1); 
transitionNetState.outs=outDegreeFactor(i,j) .* state.transitionNetState.outs /sqrt_val ; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  


%weightMatrix=zeros(sqrt(sf(1)),sqrt(sf(1))*sf(2));
weightMatrix=zeros(sqrt_val,sqrt_val*sf(2));

%weightMatrix(:)=f(:);
weightMatrix(:)=transitionNetState.outs(:);


% state.tranMatrix=extendedKron(dataSet.(dataset).connMatrix,weightMatrix,sqrt(sf(1)));
state.tranMatrix=extendedKron(dataset,weightMatrix,sqrt_val);

%[state.forcingFactor,state.forcingNet]=feval(sys.forcingNet.forwardFunction,dataSet.(dataset).nodeLabels,p.forcingNet);
[state.forcingNet]=feval(dynamicSystem.config.forcingNet.forwardFunction,dataSet.(dataset).nodeLabels,'forcingNet',optimalParam);
%E=state.forcingFactor(:);
E=state.forcingNet.outs(:);

y=x(:);
for i=1:maxIt
    ny=state.tranMatrix *y+E;

    stabCoef=(sum(sum(abs(ny-y)))) / sum(sum(abs(y)));
    y=ny;
    if(stabCoef<learning.config.forwardStopCoefficient)
        break;
    end

end

r=zeros(size(x));
r(:)=y(:);
