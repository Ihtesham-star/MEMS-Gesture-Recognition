function startTcpServer()
    global t mMTIDevice;
    
    % Set this flag to true for laser to shine directly ON your fingertip
    fingertip_targeting_mode = true;
    
    % Constants for motion control
    if fingertip_targeting_mode
        SMOOTHING_FACTOR = 0.7;  % More smoothing for fingertip targeting
    else
        SMOOTHING_FACTOR = 0.3;  % Original smoothing factor
    end
    prev_x = 0;
    prev_y = 0;
    
    % For fingertip targeting mode
    prev_target_x = 0;
    prev_target_y = 0;
    fingertip_mode_active = false;

    % Initialize arrays for plotting
    xData = [];
    yData = [];

    % Initialize these variables
     processing_time = 0;
     deltaX = 0;
     deltaY = 0;

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
    
    if fingertip_targeting_mode
        disp('*** FINGERTIP TARGETING MODE ACTIVE ***');
        disp('Laser will shine directly ON your fingertip as you move it.');
    end

    % Setup visualization
    fig = figure('Name', 'Finger Tracking Visualization');
    hold on;
    axis([0 1920 0 1080]);
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    if fingertip_targeting_mode
        title('Real-time Index Fingertip Targeting (Laser on Fingertip)');
    else
        title('Real-time Index Fingertip Tracking');
    end
    grid on;
    h = plot(nan, nan, 'b-o', 'LineWidth', 1.5);
    rectangle('Position', [860 480 200 120], 'EdgeColor', 'r', 'LineStyle', '--');

    % Open timing data log file
    fileID = fopen('laser1_response.csv', 'w');
    if fileID == -1
        error('Failed to open file for writing.');
    end
    fprintf(fileID, '%s\n', ['Timestamp,Reception_Time,Processing_Time,Control_Time,' ...
                            'Laser_X,Laser_Y,Target_X,Target_Y,Square_Mode,' ...
                            'Square_Width,Square_Height,Normalized_X,Normalized_Y']);

    % Variables for square mode
    is_square_mode = false;
    square_mode_x = 0;
    square_mode_y = 0;
    square_width = 0;
    square_height = 0;
    received_dimensions = false;
    control_time = 0;  % Initialize control_time
    
     % Variables for square mode
is_square_mode = false;
square_mode_x = 0;
square_mode_y = 0;
square_width = 0;
square_height = 0;
received_dimensions = false;
control_time = 0;  % Initialize control_time
processing_time = 0;  % Initialize processing_time
deltaX = 0;  % Initialize deltaX
deltaY = 0;  % Initialize deltaY

    % Main processing loop
    while true
        if t.NumBytesAvailable >= 16
            loop_start_time = posixtime(datetime('now'));
            
            data = read(t, 2, 'double');
            reception_time = posixtime(datetime('now')) - loop_start_time;
            
            x = data(1);
            y = data(2);
            
            % Check if this is a special fingertip targeting message
            if y == -9999
                % This is a flag for fingertip targeting
                % Wait for the real coordinates
                while t.NumBytesAvailable < 16
                    pause(0.001);
                end
                data = read(t, 2, 'double');
                x = data(1);
                y = data(2);
                
                % Calculate normalized coordinates for targeting
                target_x = (x - 960) / 960;
                target_y = (1080 - y - 540) / 540;
                
                % Apply strong smoothing
                target_x = prev_target_x * 0.8 + target_x * 0.2;
                target_y = prev_target_y * 0.8 + target_y * 0.2;
                prev_target_x = target_x;
                prev_target_y = target_y;
                
                % Send command to the MEMS device
                control_start = posixtime(datetime('now'));
                mMTIDevice.SendDataStream(target_x, target_y, 0, 1, 1, 1);
                control_time = posixtime(datetime('now')) - control_start;
                
                % Use green for targeting mode
                set(h, 'Color', 'g');
                fingertip_mode_active = true;
            else
                % Apply normal smoothing
                x = prev_x * SMOOTHING_FACTOR + x * (1 - SMOOTHING_FACTOR);
                y = prev_y * SMOOTHING_FACTOR + y * (1 - SMOOTHING_FACTOR);
                prev_x = x;
                prev_y = y;
                
                processing_start = posixtime(datetime('now'));
                
                % Direct normalization without angle limits
                deltaX = (x - 960) / 960;
                deltaY = (1080 - y - 540) / 540;
                
                % By default, send the position using standard method
                if ~fingertip_targeting_mode || ~fingertip_mode_active
                    control_start = posixtime(datetime('now'));
                    mMTIDevice.SendDataStream(deltaX, deltaY, 0, 1, 1, 1);
                    control_time = posixtime(datetime('now')) - control_start;
                else
                    control_time = 0;  % Set to 0 if not calculated yet
                end
                
                processing_time = posixtime(datetime('now')) - processing_start;
                
                % Record timing data with position error calculation
                if ~is_square_mode
                    position_error = sqrt((x - data(1))^2 + (y - data(2))^2);
                    response_delay = control_time + processing_time + reception_time;
                end
            end
            
            % Write data to file
            fprintf(fileID, '%d,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.6f,%.6f\n', ...
                loop_start_time, ...
                reception_time, ...
                processing_time, ...
                control_time, ...
                x, y, ...
                data(1), data(2), ...
                is_square_mode, ...
                square_width, ...
                square_height, ...
                deltaX, deltaY);
            
            % Update visualization
            xData = [xData, x];
            yData = [yData, 1080 - y];
            set(h, 'XData', xData, 'YData', yData);
            drawnow limitrate;

            if t.NumBytesAvailable > 0
                command = char(read(t, t.NumBytesAvailable, 'char'));
                
                if contains(command, 'dimensions')
                    % Store the received dimensions
                    square_width = data(1);
                    square_height = data(2);
                    received_dimensions = true;
                    
                elseif contains(command, 'move_to_start')
                    % Get the target position from the previous data read
                    target_x = (data(1) - 960) / 960;
                    target_y = (1080 - data(2) - 540) / 540;
                    
                    % Current position
                    current_x = (x - 960) / 960;
                    current_y = (1080 - y - 540) / 540;
                    
                    % Smooth transition
                    steps = 20;
                    for i = 1:steps
                        t_step = i/steps;
                        interpX = current_x + (target_x - current_x) * t_step;
                        interpY = current_y + (target_y - current_y) * t_step;
                        mMTIDevice.SendDataStream(interpX, interpY, 0, 1, 1, 1);
                        pause(0.01);
                    end
                    
                elseif contains(command, 'pinch') && ~is_square_mode && received_dimensions
                    is_square_mode = true;
                    square_mode_x = x;
                    square_mode_y = y;
                    
                    % Calculate normalized coordinates
                    center_x_norm = (square_mode_x - 960) / 960;
                    center_y_norm = (1080 - square_mode_y - 540) / 540;
                    
                    % Calculate normalized dimensions
                    half_width = (square_width / 2) / 960;
                    half_height = (square_height / 2) / 540;
                    
                    % Define square coordinates
                    squareCoords = [
                        center_x_norm - half_width, center_y_norm + half_height;
                        center_x_norm + half_width, center_y_norm + half_height;
                        center_x_norm + half_width, center_y_norm - half_height;
                        center_x_norm - half_width, center_y_norm - half_height;
                        center_x_norm - half_width, center_y_norm + half_height
                    ];
                    
                    % Draw the square with improved interpolation
                    for i = 1:size(squareCoords, 1)-1
                        steps = 20;
                        for j = 1:steps
                            interpX = squareCoords(i,1) + (squareCoords(i+1,1) - squareCoords(i,1)) * (j/steps);
                            interpY = squareCoords(i,2) + (squareCoords(i+1,2) - squareCoords(i,2)) * (j/steps);
                            
                            mMTIDevice.SendDataStream(interpX, interpY, 0, 1, 1, 1);
                            pause(0.01);
                        end
                    end
                    
                elseif contains(command, 'release')
                    is_square_mode = false;
                    received_dimensions = false;
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