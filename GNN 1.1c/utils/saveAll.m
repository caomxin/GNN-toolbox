function saveAll(fname)

global dynamicSystem learning testing

if nargin==0
    fname=['/tmp/' dynamicSystem.config.name '.mat'];
end
try
    save(fname, 'dynamicSystem','learning','testing');
catch
    rethrow(lasterror)
end
