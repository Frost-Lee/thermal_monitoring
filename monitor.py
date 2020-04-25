import cv2
import argparse
import h5py
import os
import numpy as np

import thermal_camera as tc

arg_parser = argparse.ArgumentParser(
    description='Real time & multiple people body temperature and repository rate estimation with thermal imaging.'
)
arg_group = arg_parser.add_mutually_exclusive_group(required=True)
arg_group.add_argument(
    '-r',
    '--record',
    nargs=2,
    help='Record the captured frames as a HDF5 file. Arguments: GIGE thermal camera ID; file path.'
)
arg_group.add_argument(
    '-e',
    '--estimate',
    type=str,
    help='Estimate and visualize the body temperature and repository rate. Argument: GIGE thermal camera ID or record file path.'
)
args = arg_parser.parse_args()

if args.record:
    from thermal_monitor import utils
    with h5py.File(args.record[1], 'w') as out_file:
        frame_index = 0
        print('Recording. Press Ctrl + C to stop.')
        for raw_frame, timestamp in tc.data_feed.stream_feed(args.record[0]):
            out_file['frame_{}/raw_frame'.format(frame_index)] = raw_frame
            out_file['frame_{}/timestamp'.format(frame_index)] = np.array([timestamp])
            frame_index += 1
            cv2.imshow('thermal recording', utils.rescale(raw_frame))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    exit()

if args.estimate:
    import thermal_monitor as tm
    if os.path.exists(args.estimate):
        feed = tc.data_feed.file_feed(args.estimate)
    else:
        feed = tc.data_feed.stream_feed(args.estimate)
    print('Visualizing estimation result. Press Ctrl + C to stop.')
    visualizer = tm.visualizer.Visualizer()
    visualizer.run(feed)
    exit()
