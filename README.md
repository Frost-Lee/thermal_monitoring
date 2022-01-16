# Thermal Monitoring

Real time & multiple people body temperature and repository rate estimating with thermal imaging.

![Demo](https://tsanchenli.com/s2/vF9F6qYQcEo3Sx7zwtAy7L.gif)

### Dependencies

- MATLAB required
    - Image Acquisition Toolbox
    - Image Acquisition Toolbox Support Package for GigE Vision Hardware
- Python packages
    - insightface
    - opencv-python
    - mxnet-cu90mkl (GPU) or mxnet-mkl (CPU)
    - etc
- GIGE thermal camera (tested on [FLIR A35](https://www.flir.com/products/a35/))

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
- As far as we have observed, the temperature estimation converge to the environment temperature as the distance to the camera grows. Thus for testing the body temperature, one should stay close to the camera.

### See Also

https://github.com/boston-dynamics/bosdyn-hospital-bot

