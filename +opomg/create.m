function p = create(op,l1,l2,lz,name,K)

if nargin < 6
    K=1;
end

if nargin < 5
    name='';
end

if nargin < 3
    l2 = [];
end
if nargin < 4
    lz = [];%any
end
if isempty(l2)
    l2 = l1;
end
l2=l2(:);
l1=l1(:);
lz=lz(:);
if K > 1
    commutative = strcmp(op,'+') || strcmp(op,'*');
    p={};
    p1 =fix(linspace(1,length(l1),K+1));
    p2 =fix(linspace(1,length(l2),K+1));
    for I=1:length(p1)-1
        if commutative
            J1=I;
        else
            J1=1;
        end
        for J=J1:length(p2)-1
            ename = [name sprintf('split%dx%d',I,J)];
            al1 = l1(p1(I):p1(I+1));
            al2 = l2(p2(J):p2(J+1));
            ps = opomg.create(op,al1,al2,lz,ename,1);
            p=[p;ps];
        end
    end
    return;
end

p = [];
p.l1 = l1;
p.l2 = l2;
p.lz =lz;
p.op = op;
p.type = 'omgop';
p.mono = false;
commutative=false;

switch(op)
    case '+'
        %l = [-l; 0; l];
        fx = @(x,y) x+y;    
        commutative = true;
        p.mono = true;
    case '-'
        % only positive
        fx = @(x,y) x-y;
        p.mono = true;
        p.negative = true;
    case '*'
        % only positive
        fx = @(x,y) x.*y;
        commutative = true;
        p.mono = true;
    case '/'
        fx = @(x,y) x./y;
        p.mono = true;
        p.samex = true; % for division and atan2 without negative
        p.negative=true;
    case '^'
        fx = @(x,y) x.^y;
    case 'atan2'
        fx = @(x,y) atan2(x,y);
        p.mono = false;
        p.samex = false;
        p.negaive = true;
    otherwise
        fx =op;
        p.op = 'fx';
end
t = bsxfun(fx,l1,l2');
t = real(t); 
if isempty(lz) == 0
    tw = t(:); % values of the operation
    idzero = find(lz==0,1,'first');     % where is Zero?   
    idnan = find(isnan(lz)); % where is nan?
    lzi = 1:length(lz); % all indices
    if isempty(idzero) == 0
        lzi = setdiff(lzi,idzero(1)); % remove zero
    end
    if isempty(idnan) == 0
        lzi = setdiff(lzi,idnan); % remove nan indices
    end
    lzz = lz(lzi); % these are output values for the comparison
    
    [reo,~,uii] = unique(tw); % identify unique values, and their mapping to tw


    reo_to_z= ones(length(reo),1); % prepare output (1 for ignarble)
    

    % could optimize this
    for I=1:length(reo)
        [~,zi]=min(abs(reo(I)-lzz)); % nearest in good group
        
        reo_to_z(I) = zi;       % value 
    end
    % then map every 
    two = lzz(reo_to_z(uii)); % original index to new index in lz
    
    

    % replace zeros
    if isempty(idzero) == 0
        two(tw == 0) = 0;
    end
    if isempty(idnan) == 0
        two(isnan(tw)) = nan;
    end
    
    t = reshape(two,size(t));

    
end
p.t = t;
if nargin < 5
    p.name = op;
else
    p.name = name
end
p.ename=name;
p.commutative=commutative;



