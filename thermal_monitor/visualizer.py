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
        self.ax_cache = []
    
    def run(self, feed, visualize_temperature=True, visualize_breath_rate=True, visualize_breath_curve=True):
        if visualize_breath_curve:
            self._visualize_breath_curves()
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
    
    def _visualize_breath_curves(self):
        self.breath_curve_figure = plt.figure()
        def update_curves(frame):
            if len(self.thermal_frame_queue) == 0:
                return
            for ax in self.ax_cache:
                ax.remove()
            self.ax_cache = []
            for index, face in enumerate(self.thermal_frame_queue[-1].thermal_faces):
                ax = self.breath_curve_figure.add_subplot(
                    len(self.thermal_frame_queue[-1].thermal_faces),
                    1,
                    index + 1,
                    label=face.uuid
                )
                self.ax_cache.append(ax)
                ax.clear()
                ax.plot(*face.breath_samples)
        ani = animation.FuncAnimation(self.breath_curve_figure, update_curves, interval=10)
        plt.show(block=False)
