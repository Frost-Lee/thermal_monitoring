function gigecamHandle = init_gigecam(serialNum)

if nargin == 0
    % if serial number is not set, set a default value
    serialNum = '73300278';
end

% create the gigecam object
gigecamHandle = gigecam(serialNum);

% set temperature mode
set(gigecamHandle,'TemperatureLinearMode','On');
set(gigecamHandle,'TemperatureLinearResolution','High');