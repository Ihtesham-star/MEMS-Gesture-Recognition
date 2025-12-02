function startTcpServer()
    global t mMTIDevice;
    
    % Constants
    SMOOTHING_FACTOR = 0.3;
    prev_x = 0;
    prev_y = 0;

    % Server setup with error handling
    if exist('t', 'var') && isa(t, 'tcpserver') && isvalid(t)
        delete(t);
        clear t;
    end
    t = tcpserver('0.0.0.0', 30001);
    disp(['Server running on port ', num2str(t.LocalPort)]);

    % MEMS device initialization
    mMTIDevice = MTIDevice();
    availableDevices = mMTIDevice.GetAvailableDevices();
    disp('Available devices:');
    disp(availableDevices);

    if availableDevices.NumDevices > 0
        mMTIDevice.ConnectDevice(availableDevices.CommPortName{1});
        disp(['Successfully connected to MEMS device on ', availableDevices.CommPortName{1}]);
    else
        error('No MEMS devices found. Please check physical connections.');
    end

    % Configure MEMS device
    mMTIDevice.SetDeviceParam(MTIParam.Vbias, 90);
    mMTIDevice.SetDeviceParam(MTIParam.MEMSDriverEnable, 1);
    mMTIDevice.SetDeviceParam(MTIParam.LaserModulationEnable, 1);
    disp('MEMS device configuration completed.');

    % Setup visualization
    fig = figure('Name', 'Laser Position Visualization');
    hold on;
    xlim([-1 1]);
    ylim([-1 1]);
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    title('Real-time Laser Position');
    grid on;
    h = plot(nan, nan, 'b-o', 'LineWidth', 1.5);
    
    % Create indicator for grab state
    grabText = text(0.7, 0.9, 'State: Idle', 'FontSize', 14);

    % Open timing data log file
    fileID = fopen('laser_simulation_response.csv', 'w');
    if fileID == -1
        error('Failed to open file for writing.');
    end
    fprintf(fileID, '%s\n', ['Timestamp,Reception_Time,Processing_Time,Control_Time,' ...
                            'Laser_X,Laser_Y,Target_X,Target_Y,Mode']);

    % Variables for state management
    is_grabbing = false;
    is_drawing_pattern = false;

    % Limited size buffer for position history
    max_history = 100;
    xData = zeros(1, max_history);
    yData = zeros(1, max_history);
    history_index = 1;

    % Main processing loop
    while true
        if t.NumBytesAvailable >= 16
            loop_start_time = posixtime(datetime('now'));
            
            % Read position data
            data = read(t, 2, 'double');
            reception_time = posixtime(datetime('now')) - loop_start_time;
            
            x = data(1);
            y = data(2);
            
            % Apply smoothing
            x = prev_x * SMOOTHING_FACTOR + x * (1 - SMOOTHING_FACTOR);
            y = prev_y * SMOOTHING_FACTOR + y * (1 - SMOOTHING_FACTOR);
            prev_x = x;
            prev_y = y;
            
            processing_start = posixtime(datetime('now'));
            
            % Convert simulation coordinates to MEMS mirror angles
            % Assuming simulation coordinates are in range [-1, 1]
            deltaX = x;  % Already normalized in simulation
            deltaY = y;  % Already normalized in simulation
            
            control_start = posixtime(datetime('now'));
            mMTIDevice.SendDataStream(deltaX, deltaY, 0, 1, 1, 1);
            control_time = posixtime(datetime('now')) - control_start;
            
            processing_time = posixtime(datetime('now')) - processing_start;
            
            % Update position history
            xData(history_index) = x;
            yData(history_index) = y;
            history_index = mod(history_index, max_history) + 1;
            
            % Write data to file
            if is_grabbing
    modeStr = 'GRAB';
else
    modeStr = 'MOVE';
end

fprintf(fileID, '%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n', ...
                loop_start_time, ...
                reception_time, ...
                processing_time, ...
                control_time, ...
                x, y, ...
                data(1), data(2), ...
                modeStr);

            
            % Update visualization - only plot non-zero values
            valid_indices = find(xData ~= 0 | yData ~= 0);
            set(h, 'XData', xData(valid_indices), 'YData', yData(valid_indices));
            drawnow limitrate;

            % Check for additional commands
            if t.NumBytesAvailable > 0
                command = char(read(t, t.NumBytesAvailable, 'char'));
                
                if contains(command, 'pinch')
                    is_grabbing = true;
                    set(grabText, 'String', 'State: Grabbing', 'Color', 'r');
                    
                    % Optional: Increase laser intensity or pattern for grab
                    % This would depend on your specific MEMS device capabilities
                    
                elseif contains(command, 'release')
                    is_grabbing = false;
                    set(grabText, 'String', 'State: Idle', 'Color', 'b');
                    
                    % Optional: Reset laser intensity
                    
                elseif contains(command, 'draw')
                    % Handle special drawing commands
                    % For future extensions
                end
            end
        end
        pause(0.001);
    end
end

function cleanServer()
    global t mMTIDevice;
    if exist('mMTIDevice', 'var') && isa(mMTIDevice, 'MTIDevice') && isvalid(mMTIDevice)
        mMTIDevice.StopDataStream();
        mMTIDevice.ResetDevicePosition();
        mMTIDevice.SetDeviceParam(MTIParam.MEMSDriverEnable, 0);
        mMTIDevice.SetDeviceParam(MTIParam.LaserModulationEnable, 0);
        mMTIDevice.DisconnectDevice();
        delete(mMTIDevice);
        clear mMTIDevice;
    end
    if exist('t', 'var') && isa(t, 'tcpserver') && isvalid(t)
        delete(t);
        clear t;
    end
    disp('Cleanup completed successfully.');
end