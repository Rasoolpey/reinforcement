function run_simulations(model, ips)
    load_system(model);

    send_block = [model, '/TCPSend'];
    receive_block = [model, '/TCPReceive'];

    simIn = repmat(Simulink.SimulationInput(model), 1, length(ips));
    simIn(1) = simIn(1).setBlockParameter(send_block, 'Host', '127.0.0.100');
    simIn(1) = simIn(1).setBlockParameter(receive_block, 'Host', '127.0.0.100');
    simIn(2) = simIn(2).setBlockParameter(send_block, 'Host', '127.0.0.101');
    simIn(2) = simIn(2).setBlockParameter(receive_block, 'Host', '127.0.0.101');
    simIn(3) = simIn(3).setBlockParameter(send_block, 'Host', '127.0.0.102');
    simIn(3) = simIn(3).setBlockParameter(receive_block, 'Host', '127.0.0.102');
    simIn(4) = simIn(4).setBlockParameter(send_block, 'Host', '127.0.0.103');
    simIn(4) = simIn(4).setBlockParameter(receive_block, 'Host', '127.0.0.103');
    simIn(5) = simIn(5).setBlockParameter(send_block, 'Host', '127.0.0.104');
    simIn(5) = simIn(5).setBlockParameter(receive_block, 'Host', '127.0.0.104');
    simIn(6) = simIn(6).setBlockParameter(send_block, 'Host', '127.0.0.105');
    simIn(6) = simIn(6).setBlockParameter(receive_block, 'Host', '127.0.0.105');
    simIn(7) = simIn(7).setBlockParameter(send_block, 'Host', '127.0.0.106');
    simIn(7) = simIn(7).setBlockParameter(receive_block, 'Host', '127.0.0.106');
    simIn(8) = simIn(8).setBlockParameter(send_block, 'Host', '127.0.0.107');
    simIn(8) = simIn(8).setBlockParameter(receive_block, 'Host', '127.0.0.107');
    simIn(9) = simIn(9).setBlockParameter(send_block, 'Host', '127.0.0.108');
    simIn(9) = simIn(9).setBlockParameter(receive_block, 'Host', '127.0.0.108');
    simIn(10) = simIn(10).setBlockParameter(send_block, 'Host', '127.0.0.109');
    simIn(10) = simIn(10).setBlockParameter(receive_block, 'Host', '127.0.0.109');
    % Run simulations and capture output
    simOut = parsim(simIn, 'ShowProgress', 'on');

    % Check for errors in simulation outputs
    errors = cell(1, length(simOut));
    for i = 1:length(simOut)
        if simOut(i).ErrorMessage
            errors{i} = simOut(i).ErrorMessage;
        else
            errors{i} = 'No error';
        end
    end

    % Save errors to a file or print them
    save('simulation_errors.mat', 'errors');
    disp(errors);
end
