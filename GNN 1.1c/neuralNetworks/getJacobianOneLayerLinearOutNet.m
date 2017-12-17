function [gradient,dInputs]=getJacobianOneLayerLinearOutNet(net,netState,delta,jacNode)
exampleNum=size(netState.inputs,2);
hiddensNum=size(net.weights1,1);
outNum=size(delta,1);

hiddenFirstDer=1-netState.hiddens.*netState.hiddens;
hiddenSecondDer=-2*netState.hiddens .* (1-netState.hiddens.*netState.hiddens);
gradient.weights1=delta*(netState.hiddens .* repmat(net.weights1(:,jacNode),hiddensNum,exampleNum))';
gradient.bias1=zeros(outNum,1);
dnetf1=(net.weights2'*delta) .* hiddenSecondDer .*repmat(net.weights1(:,jacNode),hiddensNum,exampleNum);
df2=(net.weights2'*delta) .* hiddenFirstDer ;
dnetf2=sparse([],[],[],hiddensNum,exampleNum);
dnetf2(:,jacNode)=df2;
gradient.weights1=dnetf1 * netState.inputs';
gradient.bias1=sum(dnetf1,2);    %sum(dnetf1')'



