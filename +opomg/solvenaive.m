function r = solvenaive(problem)

if iscell(problem)
    r={};
    for I=1:length(problem)
        r=[r;opomg.solvenaive(problem{I})];
    end
    return;
end

problem.fx = [];
if isfield(problem,'mono') == 0
    problem.mono = false;
end
if isfield(problem,'first0') == 0
    problem.first0 = false;
end
tic

e = toc;
r=[];
r.cmd='naive';
r.elapsed  = e;
r.type = 'omgsolution';
r.Lq = 0;
r.samex=false;
r.p=problem.p;
r.negative = false;

% solve the problem by the naive model
r.Lx1 = (1:problem.nx1)'*problem.nx2;
r.Lx2 = (1:problem.nx2)';
r.Ly = (1:(r.Lx1(end)+r.Lx2(end)))';
r.iLy = sparse(max(r.Ly(:))+1,1);
r.iLy(r.Ly(:)+1) = r.Ly;
% redefine b that is the index to  
%UNMAP r.b = r.iLy(r.Lx1(problem.p(:,1))+r.Lx2(problem.p(:,2)));

r.b = (r.Lx1(problem.p(:,1))+r.Lx2(problem.p(:,2)));

% 
% then the values

% Lq
% Lx1
% Lx2
% Ly
r.solved = true;
