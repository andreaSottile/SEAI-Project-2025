function r = positlist(n,k)

if k > 2
    r=[];
     return;
else
switch(n)
    case 3
        assert(k==0);
        r =[0.5 1 2]';
        % NaR and 0
    case 4
        assert(k==0,'only k=0');
        r = posit4list();
    case 6
        if k == 2
            r = posit6_2list();
        else
            r = posit6list();
        end
    case 8
        if k == 1
            r = posit8_1list();
        elseif k == 2
            r = posit8_2list();
        elseif k == 0
            r = posit8list();
        else
            error('this posit is not suported');
        end        
    case 9
        if k == 0
            r = posit9_0list();
        elseif k == 2
            r = posit9_2list();
        else
            error('this posit is not suported');

        end
    case 10
        if k == 0
            r = posit10_0list();
        elseif k == 2
            r = posit10_2list();
        else
            error('this posit is not suported');

        end
    otherwise
        error('this posit is not suported');
            r = [];
            return;
end

end
