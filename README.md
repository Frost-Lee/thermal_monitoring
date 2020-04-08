# Thermal Monitoring

Real time & multiple people body temperature and repository rate estimating with thermal imaging.

![Demo](http://209.250.236.3:1910/bloghost/DsZS88qbjebkBAfj9pJXM7.gif)

### Requirement

- MATLAB required
- GIGE thermal camera (tested on [FLIR A35](https://www.flir.com/products/a35/))
- Some other python package dependencies

### Quick Start

Start a real-time body temperature and repository rate monitoring.

``` bash
python ./monitor.py -e <your camera id>
```

Record and save to file.

``` bash
python ./monitor.py -r <your camera id> "/path/to/file.hdf5"
```

Review body temperature and repository rate monitoring from a file.

``` bash
python ./monitor.py -e "/path/to/file.hdf5"
```

### Configuration

See `./thermal_face_tracker/config.py` for configuration.

### Note

- The body temperature estimation is the maximum temperature within the face bounding box, thus it could be affected by high temperature object in the background.
- The system needs to accumulate some frames for repository rate estimation. If the tracking for the face got lost, the accumulation start from beginning.