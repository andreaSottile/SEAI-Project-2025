function p = setup(pp)

if iscell(pp)
    p={};
    for I=1:length(pp)
        p=[p;opomg.setup(pp{I})];
    end
    return;
end

l1 = pp.l1;
l2 = pp.l2;
op= pp.op;
commutative= pp.commutative;
if isfield(pp,'full') == 0
    pp.full = 0;
end
name = pp.name;
t  = pp.t;
mono = pp.mono;

%%
[x1,~,itx1] = unique(l1);
assert(all(diff(x1)) > 0); % ordered
[x2,~,itx2] = unique(l2);
assert(all(diff(x2)) > 0); % ordered

x1b = itx1;
x2b = itx2;
[y,~,ity] = unique(t); % totals, andt the map from every matrix element to eacy y
    % hints
    Lx1 = [];
    Lx2 = [];
    Ly =[];

if pp.full
    c = length(x1)*length(x2); 
    p=zeros(c,2); % index pairs due to binop, also for a op a
    b=zeros(c,1);
    q = 0;
    y=zeros(c,1);
    Ly = zeros(c,1);
    Lx1 = 0:(length(x1)-1); % 0..(#X1-1)
    Lx2 = (0:length(x2)-1)*length(x1); % #X1..(#X1 #X2-1)
    for I=1:length(l1)
        for J=1:length(l2)
            
            q = q + 1;
            p(q,:)= [x1b(I),x2b(J)]; % also for combinations in Lx
            b(q) = q;
            Ly(q)= Lx1(I)+Lx2(J); % hint
            y(q) = t(I,J);
        end
    end        
    eqgroups = ity;
    b = b(1:q,:);
    p = p(1:q,:);
    Ly = Ly(1:q,:);
    ny = length(b);
    hasidentity = 1;
else
    %commutative = commutative && length(l1) == length(l2);
    assert(all(diff(y)) > 0); % ordered  
    ny = length(y);

    if commutative 
        yb = zeros(length(l1),1); % maps of xa to Ya for identity
        hasidentity = true;
        for I=1:length(l1)
            r = find(y == l1(I));
            if isempty(r)
                hasidentity = false;
                break;
            end
            yb(I) = r;
        end
        if hasidentity
            hasidentity = all(all(y(yb)==l1));
        end
    else
        hasidentity= false;
        yb = []; 
    end

    % build constraints using the
    if commutative
        c = ceil(length(l1)*(length(l2)+1)/2); % max due to symmetry
    else
        c = length(l1)*length(l2);
    end
    %A=zeros(c,ny);
    p=zeros(c,2); % index pairs due to binop, also for a op a
    b=zeros(c,1);
    % half of them!
    q=0;
    for I=1:length(l1)
        if commutative
            J0=I; % symmetry
        else
            J0=1; % all pairs
        end
        for J=J0:length(l2)
            k = sub2ind(size(t),I,J); % index in table t of product and of mapping
            q = q + 1;
    %         if hasidentity && x1b(I) == x2b(J)
    %             if y(ity(k)) == x1(x1b(I))
    %                 % skip null op
    %                 % 1*1 = 1 shall not be ha
    %                 % 0+0 = 0 shall not count
    %                 %q = q - 1;
    %                 %continue
    %             end
    %             % a+a = 2a
    %             % a*a = a^2 
    %             % a/a = 1
    %             %A(q,xb(I)) = 2;
    %         else
    %             % La+Lnull=La because they are all positive...
    %             if 0==1
    %             if x1(x1b(I)) == y(ity(k)) || x2(x1b(J)) == y(ity(k))
    %                 % skip null op
    %                 %q = q-1;
    %                 %continue
    %             end
    %             end
    %             %A(q,xb(I)) = 1;
    %             %A(q,xb(J)) = 1;
    %         end
            p(q,:)= [x1b(I),x2b(J)]; % also for combinations in Lx
            b(q) = ity(k); % in Ly
    keps= 1e-5;
            % find L(yb(I))+L(yb(J)) == L(it(k))
            if op=='+'
                assert(x1(x1b(I))+x2(x2b(J)) == y(ity(k)))
            elseif op=='*'
                assert(x1(x1b(I))*x2(x2b(J)) == y(ity(k)))
            elseif op=='/'
                assert(((x1(x1b(I))/x2(x2b(J))) - y(ity(k))) < keps);
            elseif strcmp(op,'atan2')
                assert((atan2(x1(x1b(I)),x2(x2b(J))) - y(ity(k))) < keps)
            elseif op=='^'
              %  assert(((x1(x1b(I))^x2(x2b(J))) - y(ity(k))) < keps)
            end
            % yb(I) L1 + yb(J) L2 = it(k) Lx 
            %
            % assert  A(k,:)*L = b(k)  so that   
            %   yb(I) + yb(H) = it(k)
            %   y(yb(I))+y(yb(J))=y(it(k))
            % where L is integer
        end
    end
    %A = A(1:q,:);
    b = b(1:q,:);
    p = p(1:q,:);
    eqgroups = []; % all distinct
end

%assert(all(sum(A,2) ==2))
nx1 = length(l1);
nx2 = length(l2);

problem = [];
problem.ny = ny;
problem.nx1 = nx1;
problem.nx2 = nx2;
problem.op = op;
%problem.A = A; % too big and not needed
%problem.zero = zz; % index of null
problem.p = p; % shorter
problem.b= b;
problem.eqgroups = eqgroups;
problem.commutative = commutative;
problem.samex = nx1 == nx2 && all(l1 == l2); % very important hint
problem.x1 = x1;
problem.x2 = x2;
problem.Lx1 = Lx1;
problem.Lx2 = Lx2;
problem.Ly = Ly;
problem.y = y;
problem.negative = false;
problem.app = 'solve.py';
if isfield(pp,'minint')
problem.minint = pp.minint;
else
    problem.minint=0;
end
if isfield(pp,'xpolicy')
problem.xpolicy = pp.xpolicy;
end
if isfield(pp,'ypolicy')
problem.ypolicy = pp.ypolicy;
end
problem.mono = mono; %modify before call solvee
problem.hasidentity = hasidentity; % NOT used by solver 
problem.name = name;
problem.full = pp.full;
problem.type = 'omgproblem';
if isfield(pp,'negative')
    problem.negative = pp.negative;
end
if isfield(pp,'maxint')
    problem.maxint = pp.maxint;
end
p = problem;
