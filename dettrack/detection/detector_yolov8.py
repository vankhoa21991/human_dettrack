from ultralytics import YOLO

class detectorYoloV8():
    def __init__(self, cfg):
        # loading a YOLO model
        self.model = YOLO(cfg['model'])

        # geting names from classes
        self.dict_classes = self.model.model.names
        self.class_IDS = [0] # only person
        self.conf_level = cfg['conf_level']

    def detect(self, ROI):
        y_pred = self.model.predict(ROI, conf=self.conf_level, classes=self.class_IDS, device=0, verbose=False)
        return y_pred
