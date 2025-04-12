function [solution, problem, json_sol] = solveMapping_parallel(n, k, op, genProblemOnly)
 
    % Let's define the set of values ​​for X and Y (Posit⟨4,0⟩ format)
    %n=4
    %k=0
    Nx = (2^(n - 1) - 1);
    %POSIT4 Example: X = [1/4, 1/2, 3/4, 1, 3/2, 2, 4];
    X = positlist(n, k)';
    Y = X; % In this case X and Y are the same
    n = length(X);  % n = 7
    
    ptab = bsxfun(op, X', X);
    cloptab = closestPtab(ptab, X');
    
    
    % For division, the objective function is f(x,y) = x / y, which is increasing with respect to x and decreasing with respect to y.
    % The mapping we want to obtain is:
    %   x ÷ y = fz( Lx(x) + Ly(y) )

    % Decision variables: the vector x_opt = [Lx(1); ...; Lx(n); Ly(1); ...; Ly(n)]
    numVars = 2 * n;

    % Goal: minimize the overall sum ∑ Lx + ∑ Ly
    f = ones(numVars, 1);

    % We set the lower and upper bounds for the variables (natural numbers)
    lb = zeros(numVars, 1);
    ub = Inf(numVars, 1);

    A = [];
    b = [];
    Aeq = [];
    beq = [];
    rowIndex = 1;   % Contatore per le righe aggiunte

    % Constraint 1: Lx must be increasing: for i = 1,...,n-1,
    %    Lx(i+1) - Lx(i) >= 1   =>   -Lx(i+1) + Lx(i) <= -1
    for i = 1:(n - 1)
        row = zeros(1, numVars);
        row(i) = 1;
        row(i + 1) = -1;
        A = [A; row];
        b = [b; -1];
        rowIndex = rowIndex + 1;
    end

    % Constraint 2: varies depending on the arithmetic operation I'm dealing with
    if isequal(op, @rdivide) || isequal(op, @minus)
        % subtraction-division
        % Ly must be decreasing: for j = 1,...,n-1,
        % Ly(j) - Ly(j+1) >= 1   =>   - Ly(j) + Ly(j+1) <= -1
        % Ly variables are in array positions n+1 ... 2*n
        for j = 1:(n - 1)
            row = zeros(1, numVars);
            row(n + j) = -1;
            row(n + j + 1) = 1;
            A = [A; row];
            b = [b; -1];
            rowIndex = rowIndex + 1;
        end
        disp("Inverted constraints set");
    elseif isequal(op, @times) || isequal(op, @plus)
        % addition - multiplication
        % Ly must be increasing: for j = 1,...,n-1,
        % -Ly(j) + Ly(j+1) >= 1   =>   + Ly(j) - Ly(j+1) <= -1
        % Ly variables are in array positions n+1 ... 2*n
        for j = 1:(n - 1)
            row = zeros(1, numVars);
            row(n + j) = 1;
            row(n + j + 1) = -1;
            A = [A; row];
            b = [b; -1];
            rowIndex = rowIndex + 1;
        end
        disp("Monotonic constraints set");
    end

    % Start the worker pool (if it is not already active)
    if isempty(gcp('nocreate'))
        parpool; % Start a worker pool with the number of available cores
    end

    % Initializes a cell array to collect local results
    numWorkers = n; % Number of workers (one for each iteration of i)
    tempA_cell = cell(numWorkers, 1); % Cell for A
    tempB_cell = cell(numWorkers, 1); % Cell for b
    
    % Constraint 3 - Global Contraints: 
    % for each couple (i, j) and (p, q) such that
    %    X(i)/Y(j) < X(p)/Y(q)
    % we impose:
    %    Lx(i) + Ly(j) + 1 <= Lx(p) + Ly(q)
    % So: Lx(i) + Ly(j) - Lx(p) - Ly(q) <= -1
    parfor i = 1:n
        tempA = []; % Temporary matrix for A (local to each worker)
        tempB = []; % Temporary matrix for b (local to each worker)
        
        for j = 1:n
            if j > i
                continue; % Skip iterations where j > i
            end
            for p = 1:n
                if p == i
                    continue; % Skip iterations where p == i
                end
                for q = 1:n
                    if isequal(op, @times)
                        if q>p
                            continue; % Skip iterations where q > p 
                        end
                        if X(i) * Y(j) < X(p) * Y(q)
                            row = zeros(1, numVars);
                            row(i) = 1;          % Lx(i)
                            row(n + j) = 1;      % Ly(j)
                            row(p) = -1;         % Lx(p)
                            row(n + q) = -1;     % Ly(q)
                            if i == p
                                row(p) = 0;
                            elseif j == q
                                row(n + q) = 0;
                            end
                            tempA = [tempA; row];
                            tempB = [tempB; -1];
                        end
                    elseif isequal(op, @plus)
                        if q > p
                            continue; % Skip iterations where q > p
                        end
                        if X(j) + Y(i) < X(p) + Y(q)
                            row = zeros(1, numVars);
                            row(i) = 1;          % Lx(i)
                            row(n + j) = 1;      % Ly(j)
                            row(p) = -1;         % Lx(p)
                            row(n + q) = -1;     % Ly(q)
                            if i == p
                                row(p) = 0;
                            elseif j == q
                                row(n + q) = 0;
                            end
                            tempA = [tempA; row];
                            tempB = [tempB; -1];
                        end
                    elseif isequal(op, @rdivide)
                        if q>p
                            continue; % Skip iterations where q > p 
                        end
                        if X(i) / Y(j) < X(p) / Y(q)
                            row = zeros(1, numVars);
                            row(i) = 1;          % Lx(i)
                            row(n + j) = 1;      % Ly(j)
                            row(p) = -1;         % Lx(p)
                            row(n + q) = -1;     % Ly(q)
                            if i == p
                                row(p) = 0;
                            elseif j == q
                                row(n + q) = 0;
                            end
                            tempA = [tempA; row];
                            tempB = [tempB; -1];
                        end
                    elseif isequal(op, @minus)
                        if q>p
                            continue; % Skip iterations where q > p 
                        end
                        if X(i) - Y(j) < X(p) - Y(q)
                            row = zeros(1, numVars);
                            row(i) = 1;          % Lx(i)
                            row(n + j) = 1;      % Ly(j)
                            row(p) = -1;         % Lx(p)
                            row(n + q) = -1;     % Ly(q)
                            if i == p
                                row(p) = 0;
                            elseif j == q
                                row(n + q) = 0;
                            end
                            tempA = [tempA; row];
                            tempB = [tempB; -1];
                        end
                    end
                end
            end
        end
        
        % Store local results in cells
        tempA_cell{i} = tempA;
        tempB_cell{i} = tempB;
    end
    
    % Merge the local results in A and b
    rowIndex = 1;
    for i = 1:numWorkers
        if ~isempty(tempA_cell{i})
            numRows = size(tempA_cell{i}, 1);
            A = [A; tempA_cell{i}];
            b = [b; tempB_cell{i}];
            rowIndex = rowIndex + numRows;
        end
    end

    % Equality Constraints
    if isequal(op, @times) || isequal(op, @plus)
        for r = 2:Nx
            for c = 1:r - 1
                constrRow = zeros(numVars, 1)';
                constrRow(r) = 1;
                constrRow(c + Nx) = 1;
                constrRow(c) = -1;
                constrRow(r + Nx) = -1;
                Aeq = [Aeq; constrRow];
                beq = [beq; 0];
            end
        end
        disp("Equality constraints set");
    end

    % Transform int8 A into sparse double A
    disp("[TRANSFORMING INT8 MATRIX INTO SPARSE DOUBLE]")
    Al = sparse(logical(uint8(A)));
    An = -A;
    clear A;
    Aln = sparse(logical(uint8(An)));
    clear An;
    Ad = double(Al) - double(Aln);
    clear Al;
    clear Aln;
    A = Ad;
    clear Ad;
    clear tempA;
    clear tempB;
    clear i_idx;
    clear j_idx;
    clear condition;
    clear condition2;
    clear prod_ij;
    clear prod_pq;
    clear q_idx;
    clear numRows;
    clear row;

    % Problem configuration
    disp("Problem setup completed");
    % All variables are integers
    intcon = 1:numVars;
    options = optimoptions('intlinprog', 'Display', 'iter');
    opts = optimoptions('intlinprog', 'MaxTime', 100000);
    problem.f = f;
    problem.intcon = intcon;
    problem.A = A;
    problem.b = b;
    problem.Aeq = Aeq;
    problem.beq = beq;
    problem.lb = lb;
    problem.ub = ub;
    problem.p = X';
    problem.op = op;
    problem.optab = ptab;
    problem.cloptab = cloptab;
    clear A;
    clear Aeq;
    clear b;
    clear beq;

    disp("Solver started...");
    % Solve the problem with intlinprog
    [x_opt, fval, exitflag, output] = intlinprog(problem.f, problem.intcon, problem.A, problem.b, problem.Aeq, problem.beq, problem.lb, problem.ub, options);
    disp("Solver finished");
    % Extract the vectors Lx and Ly from the optimal solution
    Lx = int64(x_opt(1:n));
    Ly = int64(x_opt(n+1:end));
    
    % Let's build the lookup table fz:
    %   For each pair (i,j) we define the sum S = Lx(i) + Ly(j) 
    %   and then we associate with the value
    %   X(i) ∇ Y(j) (i.e. the expected result of the operation).
    Lz = zeros(n, n);
    for i = 1:n
        for j = 1:n
            Lz(i,j) = Lx(i) + Ly(j);
        end
    end
    % Extract the unique values ​​of the sum
    Lz_values = unique(Lz(:));
    % Create a lookup table using a map (containers.Map)
    fz_lookup = containers.Map('KeyType','double','ValueType','double');
    for k = 1:length(Lz_values)
        sumVal = Lz_values(k);
        % Find a pair (i,j) such that Lx(i) + Ly(j) == sumVal
        [ii, jj] = find(Lz == sumVal, 1);
        if isequal(op,@times)
            fz_lookup(sumVal) = X(ii) * Y(jj);
        elseif isequal(op,@plus)
            fz_lookup(sumVal) = X(ii) + Y(jj);
        elseif isequal(op,@minus)
            fz_lookup(sumVal) = X(ii) - Y(jj);
        elseif isequal(op,@rdivide)
            fz_lookup(sumVal) = X(ii) / Y(jj);
        end
    end
    
    % fz is defined as an anonymous function 
    % that, given the value of the sum S, 
    % returns the result 
    fz = @(S) fz_lookup(S);
 
    solution = struct;
        
    [name,sym] = getFunctionName(op);
    solution.op = name;
    solution.ophandle = op;
    solution.p  = X';
    solution.optab = ptab;
    solution.cloptab = cloptab;
    solution.Lx = Lx;
    solution.Ly = Ly;
    solution.Lz = Lz;
    solution.Lz2z = genLz2z(solution.Lz,solution.cloptab,solution.p);
    solution.verified = verify(solution.optab,solution.cloptab,solution.p,solution.Lx,solution.Ly,solution.Lz2z);
    solution.op = sym;
    json_sol = toJsonEncodedSolution(solution);   




    % Print the results
    disp('Mapping Lx (numerator operand):');
    disp(Lx');
    disp('Mapping Ly (operand denominator):');
    disp(Ly');
    disp('Lookup table (Lz -> operation result):');
    keysLz = sort(cell2mat(fz_lookup.keys));
    for k = 1:length(keysLz)
        fprintf('Lz = %d  -->  %f\n', keysLz(k), fz_lookup(keysLz(k)));
    end
end


function [name,sym] = getFunctionName(op)
    if isequal(op,@times)
        name = "mul";
        sym = "*";
    elseif isequal(op,@plus)
        name = "sum";
        sym = "+";
    elseif isequal(op,@rdivide)
        name = "div";
        sym = "/";
    %andrea
    elseif isequal(op,@minus)
        name = "min";
        sym = "-";    
    end
end



function ptab = closestPtab(optab,plist)
    r = size(optab,1);
    c = size(optab,2);
    ptab =zeros("like",optab);
    for i=1:r
        for j=1:c
            p = optab(i,j);

            % closest posit and relative index
            [~, idx] = min(abs(plist-p));
            minp = plist(idx);
            ptab(i,j) = minp;
        end
    end
end


function lz2z = genLz2z(Lz,optab,plist)
    % We map the Lz set to the original z (optab) set
    % We want to discard duplicates, since they map to the
    % same element in z
    % Elements of optab may not be present in z (round to nearest)

    r = size(optab,1);
    c = size(optab,2);
    
    lz2zKeys = [];
    lz2zVals = [];
    lz2z = struct;
    for i=1:r
        for j=1:c
            p = optab(i,j);
        
            % closest posit and relative index
            %[~, idx] = min(abs(plist-p));
            %minp = plist(idx);

            % map idx to correspondent z (only if not contained)
            lzv = Lz(i,j);
            if ~ismember(lzv,lz2zKeys)
                lz2zKeys = [lz2zKeys; lzv];
                lz2zVals = [lz2zVals; p];
            else
                Lzidx = find(lz2zKeys == lzv);
                z = lz2zVals(Lzidx);
                if double(z) ~= double(p)
                    fprintf("=====Error on %d,%d=====\n",i,j);
                    fprintf("%d already in with value: %f (new value: %f)\n",lzv,z,p );
                end
            end
        end
    end
    lz2z.keys = lz2zKeys;
    lz2z.vals = lz2zVals;
end


function verified = verify(optab,cloptab,plist,Lx,Ly,Lz2z)
    r = size(Lx,1);
    r1 = size(Ly,1);
    assert(r == r1,"Lx,Ly sizes do not match");
    verified = true;
    for i=1:r
        for j=1:r
            result = cloptab(i,j);
            Lxi = Lx(i);
            Lyj = Ly(j);
            Lzij = Lxi + Lyj;
            
            Lzidx = find(Lz2z.keys == Lzij);
            z = Lz2z.vals(Lzidx);
               
            if z ~= result 
                fprintf("===== Error on %d,%d =====\n",i,j);
                fprintf("z=%f, exp=%f, exact=%f\n",z,result,optab(i,j));
                %fprintf("x=%f, y=%f\n",plist(i),plist(j));
                %fprintf("lx=%d, ly=%d\n", Lxi,Lyj);
                %fprintf("lz=%d, lzidx=%d\n",Lzij,Lzidx);

                verified = false;
            end

        end
    end
end

function encoded = toJsonEncodedSolution(solution)
     jstruct = struct;
     jstruct.Lx1 = solution.Lx;
     jstruct.Lx2 = solution.Ly;
     jstruct.op = solution.op;
     jstruct.Ly  = reshape(solution.Lz.',1,[]);
     jstruct.uLy2y = [solution.Lz2z.keys solution.Lz2z.vals];
     jstruct.x1 = solution.p;
     jstruct.y = reshape(solution.optab.',1,[]);
        
     encoded = jsonencode(jstruct);
     
end
