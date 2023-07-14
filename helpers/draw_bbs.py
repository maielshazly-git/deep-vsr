import matplotlib as mpl
from matplotlib import image
from matplotlib import pyplot as plt
mpl.rcParams.update({'text.color': 'black'})

class bounding_box:
    def __init__(self, image_name, x1, y1, x2, y2, object, iou):
        self.image_name = image_name
        self.object = object
        self.iou = float(iou)
        self.x1 = round(float(x1))
        self.y1 = round(float(y1))
        self.x2 = round(float(x2))
        self.y2 = round(float(y2))

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

postfix = 'chunk{:04d}'.format(1)
bbs_file = '/240p_bbs' # '/1080p_per_frame_sr_bbs' '/1080p_per_bb_sr_bbs' '/1080p_nemo_sr_bbs'
pngs_dir = '/240p_pngs' # '/1080p_per_frame_sr_pngs' # '/1080p_per_bb_sr_pngs' # '/1080p_nemo_sr_pngs'
parent_dir = '/home/mai/nemo/data/nemo-data-traffic/traffic/detection_results/traffic/' + postfix

bbs = parse_obj_det_res(parent_dir + '/' + bbs_file)
current_img = bbs[0].image_name

for bb in bbs:
    if bb.image_name != current_img:
        plt.axis('off')
        plt.title(current_img + '.png')
        plt.imshow(image.imread(parent_dir + pngs_dir + '/' + current_img + '.png'))
        plt.show()
        current_img = bb.image_name
    # To visualize only small bbs
    #if (bb.x2 - bb.x1) * (bb.y2 - bb.y1) >= 20**2:
    #    continue
    if bb.iou < 0.5:
        continue
    plt.text(bb.x1, bb.y1, bb.object, fontsize=15)
    plt.plot([bb.x1, bb.x2], [bb.y1, bb.y1], color="red", linewidth=3) # top line
    plt.plot([bb.x2, bb.x2], [bb.y1, bb.y2], color="red", linewidth=3) # right line
    plt.plot([bb.x1, bb.x2], [bb.y2, bb.y2], color="red", linewidth=3) # bottom line
    plt.plot([bb.x1, bb.x1], [bb.y1, bb.y2], color="red", linewidth=3) # left line
