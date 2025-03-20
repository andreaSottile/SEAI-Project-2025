function pp = split(p,K)
pp={};
p1 =fix(linspace(1,length(p.l1),K+1));
p2 =fix(linspace(1,length(p.l2),K+1));
for I=1:length(p1)-1
    J1=1;
    for J=J1:length(p2)-1
        ps = p;
        if isfield(p,'ename')
            ps.ename = [ps.ename sprintf('split%dx%d',I,J)];
        else
            ps.name = [ps.name sprintf('split%dx%d',I,J)];
        end
        ps.l1 = p.l1(p1(I):p1(I+1));
        ps.l2 = p.l2(p2(J):p2(J+1));
        ps.t = p.t(p1(I):p1(I+1),p2(J):p2(J+1));
        pp=[pp;ps];
    end
end

