import cv2
from matplotlib import pyplot as plt
from matplotlib import animation

from .thermal_frame import ThermalFrame
from .thermal_face import ThermalFace
from . import config


class Visualizer(object):
    def __init__(self):
        self.thermal_frame_queue = []
        self.temperature_pool = {}
        self.breath_rate_pool = {}
        self.breath_curve_ax_pool = {}
        self.breath_curve_figure = None
        self._plot_update_counter = config.BREATH_CURVE_UPDATE_FRAMES
    
    def run(self, feed, visualize_temperature=True, visualize_breath_rate=True, visualize_breath_curve=True):
        for raw_frame, timestamp in feed:
            thermal_frame = ThermalFrame(raw_frame, timestamp)
            if len(self.thermal_frame_queue) > 0:
                thermal_frame.link(self.thermal_frame_queue[-1])
            if len(self.thermal_frame_queue) >= config.MAX_CACHED_FRAMES:
                self.thermal_frame_queue.pop(0)
                self.thermal_frame_queue[0].detach()
            self.thermal_frame_queue.append(thermal_frame)
            annotation_frame = cv2.UMat(thermal_frame.grey_frame)
            self._visualize_bounding_boxes(annotation_frame, thermal_frame.thermal_faces)
            if visualize_temperature:
                self._visualize_temperatures(annotation_frame, thermal_frame.thermal_faces)
            if visualize_breath_rate:
                self._visualize_breath_rates(annotation_frame, thermal_frame.thermal_faces)
            if visualize_breath_curve:
                self._visualize_breath_curves(thermal_frame.thermal_faces)
            cv2.imshow('thermal monitoring', cv2.resize(annotation_frame, config.VISUALIZATION_RESOLUTION))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    def _visualize_bounding_boxes(self, annotation_frame, faces):
        for face in faces:
            cv2.rectangle(
                annotation_frame, 
                tuple(face.bounding_box[:2]), 
                tuple(face.bounding_box[2:]), 
                (255, 0, 0), 
                1
            )
    
    def _visualize_temperatures(self, annotation_frame, faces):
        face_uuids = [face.uuid for face in faces]
        keys = [*self.temperature_pool.keys()]
        for key in keys:
            if key not in face_uuids:
                self.temperature_pool.pop(key, None)
        for face in faces:
            temperature = -1
            if face.uuid not in self.temperature_pool or self.temperature_pool[face.uuid][0] >= config.TEMPERATURE_UPDATE_FRAMES:
                temperature = face.temperature
                self.temperature_pool[face.uuid] = [0, temperature]
            else:
                temperature = self.temperature_pool[face.uuid][1]
                self.temperature_pool[face.uuid][0] += 1
            cv2.putText(
                annotation_frame,
                str(temperature)[:5] + ' C',
                tuple(face.bounding_box[:2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )
    
    def _visualize_breath_rates(self, annotation_frame, faces):
        face_uuids = [face.uuid for face in faces]
        keys = [*self.breath_rate_pool.keys()]
        for key in keys:
            if key not in face_uuids:
                self.breath_rate_pool.pop(key, None)
        for face in faces:
            breath_rate = -1
            if face.uuid not in self.breath_rate_pool or self.breath_rate_pool[face.uuid][0] >= config.BREATH_RATE_UPDATE_FRAMES:
                breath_rate = face.breath_rate
                if breath_rate is None:
                    return
                self.breath_rate_pool[face.uuid] = [0, breath_rate]
            else:
                breath_rate = self.breath_rate_pool[face.uuid][1]
                self.breath_rate_pool[face.uuid][0] += 1
            cv2.putText(
                annotation_frame,
                str(breath_rate)[:5] + 'Hz',
                (face.bounding_box[0], face.bounding_box[3]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )
    
    def _visualize_breath_curves(self, faces):
        if self._plot_update_counter < config.BREATH_CURVE_UPDATE_FRAMES:
            self._plot_update_counter += 1
            return
        self._plot_update_counter = 0
        if self.breath_curve_figure is None:
            self.breath_curve_figure = plt.figure()
            plt.show(block=False)
        if set([face.uuid for face in faces]) != set(self.breath_curve_ax_pool.keys()):
            for key, value in self.breath_curve_ax_pool.items():
                value.remove()
            self.breath_curve_ax_pool = {}
        for index, face in enumerate(faces):
            if face.uuid not in self.breath_curve_ax_pool:
                ax = self.breath_curve_figure.add_subplot(len(faces), 1, index + 1, label=face.uuid)
                self.breath_curve_ax_pool[face.uuid] = ax
            else:
                ax = self.breath_curve_ax_pool[face.uuid]
            ax.clear()
            ax.plot(*face.breath_samples)
        plt.draw()
        plt.pause(0.001)
