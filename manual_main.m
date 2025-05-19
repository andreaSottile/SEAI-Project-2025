%addpath('positlist')

%[a,b,e]   = genSolution(4,0,@plus)
%[a1,b1,e] = genSolution(4,0,@times);
%disp(e)
%[a1,b1,e] = genSolution(4,0,@minus);
%[s,p,e] = genSolution(6,2,@times,true);
%[a3,b4] = genSolution(6,2,@times)

%%[divs,divp,e] = genSolution(4,0,@rdivide);
[divs,divp,e] = solveMapping_speed(4,0,@minus);
saveToFile("p4_0_minus_solution.json",e)
%%andre
%[subs,subp,e] = genSolution(4,0,@minus);
%[divs,divp] = genSolution(6,2,@rdivide);

%[a1,b1,e] = genSolution(8,0,@times);
%saveToFile("p4_0_minus_solution.json",e)

function saveToFile(fname,encoded)
	id = fopen(fname,'w');
	fprintf(id,'%s',encoded);
	fclose(id);
end
