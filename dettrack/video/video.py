import cv2
from pathlib import Path
import numpy as np
class videoHuman():
    def __init__(self, cfg):
        self.scale_percent = cfg['scale_percent']

    def set_source_video(self, source):
        self.capture = cv2.VideoCapture(source)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def set_source_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def print_size(self):
        print('width, height:', self.width, self.height)

    def scale(self):
        # Scaling Video for better performance
        if self.scale_percent != 100:
            print('[INFO] - Scaling change may cause errors in pixels lines ')
            self.width = int(self.width * self.scale_percent / 100)
            self.width = int(self.height * self.scale_percent / 100)
            print('[INFO] - Dim Scaled: ', (self.width, self.height))
            return self.width, self.height
        else:
            return self.width, self.height

    def get_writer(self, output_path:Path):
        self.VIDEO_CODEC = "mp4v"

        self.writer = cv2.VideoWriter( str(output_path),
                                       cv2.VideoWriter_fourcc(*self.VIDEO_CODEC),
                                       self.fps, (self.width, self.height))
        return self.writer

    def draw_box_and_text(self, frame, box, conf, class_name, id_obj, centers_old ):
        # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
        xmin, ymin, xmax, ymax = box
        if class_name == 'human':
            color = (0, 0, 255)
        elif class_name == 'face':
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        # drawing center and bounding-box in the given frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)  # box
        for center_x, center_y in centers_old[id_obj].values():
            cv2.circle(frame, (center_x, center_y), 5, color, -1)  # center of box

        # Drawing above the bounding-box the name of class recognized.
        cv2.putText(img=frame,
                    text=id_obj + ':' + str(np.round(conf, 2)) + ' ' + class_name,
                    org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=color,
                    thickness=1)

    def stop(self):
        self.writer.release()
        cv2.destroyAllWindows()


