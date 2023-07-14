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
            if '_' not in line[0]:
                bounding_boxes.append(bounding_box(line[0], line[1], line[2], line[3], line[4], line[5], line[6]))
        f.close()
    return bounding_boxes

count_small_bb, total_len = 0, 0
for chk in range(0, 25):
    postfix = 'chunk{:04d}'.format(chk)
    bbs = parse_obj_det_res('/home/mai/nemo/data/nemo-data-traffic/traffic/detection_results/traffic/' + postfix + '/240p_bbs')
    for bb in bbs:
        if bb.score < 0.5:
            continue
        total_len += 1
        area = (bb.x2 - bb.x1) * (bb.y2 - bb.y1)
        if area < 20**2:
            count_small_bb += 1

print('Small bbs count: {}% of objects'.format(round(count_small_bb * 100 / total_len, 2)))
