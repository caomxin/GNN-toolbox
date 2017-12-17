% Augusto
% Gradiente della funzione d'errore sigmoidale

function gradErr=gradErrLogSig(alpha,m,y)

if(nargin~=3)
    error('Occorrono tre parametri: alpha,m,y');
end

if size(y,1)<size(y,2)
    errore('Attenzione y deve essere un vettore colonna');
end

sigma=logsig(alpha*m*y);
% la derivata di sigma  sigma*(1-sigma)
gradErr=(sigma.*(1-sigma))'*alpha*m;
