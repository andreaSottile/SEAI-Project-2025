function r = loadsolution(filename)
t = fileread(filename);
r = jsondecode(t);
if isempty(r)|| ~isfield(r,'Ly')
    r = [];
    r.solved = false;
else
    r.type = 'omgsolution';
    r.iLy = sparse(max(r.Ly(:))+1,1);
    r.iLy(r.Ly(:)+1) = r.Ly;
    r.solved = true;

end