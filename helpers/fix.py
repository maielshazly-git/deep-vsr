import os
import numpy as np

class bounding_box:
    def __init__(self, image_name, x1, y1, x2, y2, object, score):
        self.image_name = image_name
        self.object = object
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.score = float(score)

def parse_obj_det_res(filepath):
    with open(filepath, 'r') as f:
        bounding_boxes = []
        lines = f.readlines()
    
        for line in lines:
            line = line.split(',')
            if '_' not in line[0] and float(line[6]) >= 0.5: # filtering by confidence score 50%
                bb = bounding_box(line[0], line[1], line[2], line[3], line[4], line[5], line[6])
                if (bb.x2 - bb.x1) * (bb.y2 - bb.y1) < 20**2:
                    bounding_boxes.append(bounding_box(line[0], line[1], line[2], line[3], line[4], line[5], line[6]))
        f.close()
    return bounding_boxes

for chunk_idx in range(0, 25):
    postfix = 'chunk{:04d}'.format(chunk_idx)
    bbs = parse_obj_det_res(os.path.join('/home/mai/nemo/data/nemo-data-traffic/traffic/experiment-outputs/1-240p', postfix, '240p_bbs'))
    os.makedirs('/home/mai/nemo/data/nemo-data-traffic/traffic/experiment-outputs/1-240p-bbs/' + postfix, exist_ok=True)
    with open(os.path.join('/home/mai/nemo/data/nemo-data-traffic/traffic/experiment-outputs/1-240p-bbs', postfix, '240p_bbs'), 'a') as f:
        for bb in bbs:
            f.write(bb.image_name + ',' + str(bb.x1) + ',' + str(bb.y1) + ',' + str(bb.x2) + ',' + str(bb.y2) + ',' + bb.object + ',' + str(bb.score) + ',1.0,mpeg\n')
        f.close()
