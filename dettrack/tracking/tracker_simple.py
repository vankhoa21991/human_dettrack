from dettrack.utils.utils import risize_frame, filter_tracks, update_tracking


class Tracker():
    def __init__(self, cfg):
        self.scale_percent = cfg['scale_percent']
        self.thr_centers = cfg['thr_centers']
        self.patience = cfg['patience']
        self.alpha = cfg['alpha']
        self.frame_max = cfg['frame_max']
    def update(self, centers_old, obj_center, lastKey, i):
        centers_old, id_obj, is_new, lastKey = update_tracking(centers_old, obj_center, self.thr_centers,
                                                               lastKey, i, self.frame_max)
        return centers_old, id_obj, is_new, lastKey

    def filter_tracks(self, centers_old, patience):
        centers_old = filter_tracks(centers_old, patience)
        return centers_old