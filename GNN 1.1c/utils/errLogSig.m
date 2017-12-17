% Augusto
% Funzione d'errore sigmoidale

function err=errLogSig(alpha,m,y)

if(nargin~=3)
    error('Occorrono tre parametri: alpha,m,y');
end

if size(y,1)<size(y,2)
    error('Attenzione y deve essere un vettore colonna');
end

err=sum(logsig(alpha*m*y));





