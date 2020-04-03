function t = get_temperature(gh,row,col)

% get a snapshot from the gigecam handle
img = snapshot(gh);

% calculate temperature
if nargin == 3
    val = bitand(img(row,col),uint16(16383));
    t = double(val).*0.04-273.15;
elseif nargin == 2
    val = bitand(img(row),uint16(16383));
    t = double(val).*0.04-273.15;
else
    val = bitand(img,uint16(16383));
    t = double(val).*0.04-273.15;
end