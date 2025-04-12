function main()
    while true
        clc;
        disp('=== [DFP8 SYSTEM CONSOLE] ===');
        disp(':: Decoding-free Two-Input Arithmetic');
        disp(':: Integer Linear Optimization Interface Initialized');
        disp('----------------------------------------------------');
        disp(':: Choose which Solve Mapping version to run');
        disp('----------------------------------------------------');
        disp('1. Classic');
        disp('2. Parallel');
        disp('3. Speed');
        disp('4. Exit');
        choice = input('Select a mode: ', 's');

        switch choice
            case '1'
                promptAndRunSolvemapping('classic');
            case '2'
                promptAndRunSolvemapping('parallel');
            case '3'
                promptAndRunSolvemapping('speed');
            case '4'
                disp('Exiting...');
                break;
            otherwise
                disp('Invalid choice. Press any key to try again.');
                pause;
        end
    end
end

function promptAndRunSolvemapping(mode)
    disp(['--- Selected mode: ', mode, ' ---']);

    % Posit selection
    disp('Choose Posit type:');
    disp('1. Posit(4,0)');
    disp('2. Posit(6,0)');
    disp('3. Posit(8,0)');
    posit_choice = input('Choice: ', 's');

    switch posit_choice
        case '1'
            n = 4;
            k = 0;
        case '2'
            n = 6;
            k = 0;
        case '3'
            posit_type = 'Posit(8,0)';
            warning('Warning: Posit(8,0) may take a long time to execute.');
            pause;
            return;
        otherwise
            disp('Invalid Posit type. Returning to main menu.');
            pause;
            return;
    end

    % Operation selection
    disp('Choose the operation:');
    disp('1. Addition');
    disp('2. Subtraction');
    disp('3. Multiplication');
    disp('4. Division');
    op_choice = input('Choice: ', 's');

    switch op_choice
        case '1'
            operation = @plus;
        case '2'
            operation = @minus;
        case '3'
            operation = @times;
        case '4'
            operation = @rdivide;
        otherwise
            disp('Invalid operation. Returning to main menu.');
            pause;
            return;
    end

    % Output file name
    output_filename = input('Output JSON file name (e.g., result.json): ', "s");

    % Call the appropriate function
    switch mode
        case 'classic'
            [divs,divp,e] = solveMapping(n, k, operation);
            saveToFile(output_filename, e);
        case 'parallel'
            disp("Please wait while the Parallel Pool is configured, if no specific configuration was requested the Default profile starts");
            [divs,divp,e] = solveMapping_parallel(n, k, operation);
            saveToFile(output_filename, e)
        case 'speed'
            [divs,divp,e] = solveMapping_speed(n, k, operation);
            saveToFile(output_filename, e)
    end

    disp('Execution completed. Press any key to continue.');
    pause;
end

function saveToFile(fname,encoded)
	id = fopen(fname,'w');
	fprintf(id,'%s',encoded);
	fclose(id);
end