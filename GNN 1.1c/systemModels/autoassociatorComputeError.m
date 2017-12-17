function [e,outState]=autoassociatorComputeError(dataset,x,optimalParam)

global dataSet dynamicSystem learning

%% x will be empty except when called to test the results
if isempty(x) && strcmp(dataset,'trainSet')
    in=[dynamicSystem.state;dataSet.trainSet.nodeLabels];
elseif isempty(x) && strcmp(dataset,'validationSet')
    in=[learning.current.validationState;dataSet.validationSet.nodeLabels];
else
    in=[x;dataSet.(dataset).nodeLabels]; %%----------------gli ingressi all'autoass.
end
  

outState.outNetState=feval(dynamicSystem.config.outNet.forwardFunction,in,'outNet',optimalParam);



e=0;
e1=0;
eps=0.00001;

for i=1:size(in,2)
    %err=(.5*( outState.outNetState.outs(:,i)-in(:,i) )'*( outState.outNetState.outs(:,i) - in(:,i))+eps)^(dataSet.(dataset).targets(i));
    err=0.5*(((outState.outNetState.outs(:,i)-in(:,i))'*(outState.outNetState.outs(:,i) - in(:,i))+eps)^(dataSet.(dataset).targets(i)));
    e=e+err;
    if(dataSet.(dataset).targets(i)==1) 
        outState.delta(:,i)=(outState.outNetState.outs(:,i)-in(:,i)); 
    end
    if (dataSet.(dataset).targets(i)==-1)
        %outState.delta(:,i) = -1/(1/2*(( outState.outNetState.outs(:,i)-in(:,i) )'*( outState.outNetState.outs(:,i) - in(:,i)))+eps)^2*(outState.outNetState.outs(:,i)-in(:,i));
        outState.delta(:,i) = -(1/((outState.outNetState.outs(:,i)-in(:,i))'*(outState.outNetState.outs(:,i) - in(:,i))+eps)^2)*(outState.outNetState.outs(:,i)-in(:,i));
    end

end
