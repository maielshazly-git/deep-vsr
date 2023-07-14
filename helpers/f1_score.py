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
                bounding_boxes.append(bounding_box(line[0], line[1], line[2], line[3], line[4], line[5], line[6]))
        f.close()
    return bounding_boxes

def is_overlapping(box1, box2):
    # Reference: https://www.tutorialspoint.com/program-to-check-two-rectangular-overlaps-or-not-in-python
    if box1.x1 >= box2.x2 or box1.x2 <= box2.x1 or box1.y2 <= box2.y1 or box1.y1 >= box2.y2:
        return False
    return True

def adjust_bbs_to_scale(in_w, in_h, out_w, out_h, bb):
    width_scale = out_w /  in_w # scale of columns
    height_scale = out_h / in_h # scale of rows
    
    if bb.x1 < 1:
        bb.x1 = 1
        bb.x2 += 1
    if bb.y1 < 1:
        bb.y1 = 1
        bb.y2 += 1
            
    r1 = (bb.y1 - 1) * height_scale
    r2 = r1 - 1 + ((bb.y2 - bb.y1 + 1) * height_scale)
    c1 = (bb.x1 - 1) * width_scale
    c2 = c1 - 1 + ((bb.x2 - bb.x1 + 1) * width_scale)
    return bounding_box(bb.image_name, c1, r1, c2, r2, bb.object, bb.score)

def compute_f1_scores_for_chunk(gt_bbs_file_path, bbs_file_path, results_file_path, bbs_same_scale=True, in_w=None, in_h=None, out_w=None, out_h=None):
    f1_scores_for_frame = []
    gt_bbs_pos_in_chunk, bbs_pos_in_chunk = 0, 0
    gt_bbs, bbs = parse_obj_det_res(gt_bbs_file_path), parse_obj_det_res(bbs_file_path)
    
    while gt_bbs_pos_in_chunk < len(gt_bbs) or bbs_pos_in_chunk < len(bbs):
        gt_bbs_per_frame, bbs_per_frame = [], []
        gt_image_name, image_name = gt_bbs[gt_bbs_pos_in_chunk].image_name, bbs[bbs_pos_in_chunk].image_name
        
        # List of ground truth bounding boxes for the current frame/image in the chunk
        while gt_bbs_pos_in_chunk < len(gt_bbs) and gt_image_name == gt_bbs[gt_bbs_pos_in_chunk].image_name:
            gt_bbs_per_frame.append(gt_bbs[gt_bbs_pos_in_chunk])
            gt_bbs_pos_in_chunk += 1
        
        # List of bounding boxes to be evaluated for the current frame/image in the chunk
        while bbs_pos_in_chunk < len(bbs) and image_name == bbs[bbs_pos_in_chunk].image_name:
            if bbs_same_scale:
                bbs_per_frame.append(bbs[bbs_pos_in_chunk])
            else: # for the 240p bbs, to be able to compare them to the 1080p bbs
                scaled_bb = adjust_bbs_to_scale(in_w, in_h, out_w, out_h, bbs[bbs_pos_in_chunk])
                bbs_per_frame.append(scaled_bb)
                #print('{}\tx1:{:.2f}\ty1:{:.2f}\tx2:{:.2f}\ty2:{:.2f}'.format(image_name, bbs[bbs_pos_in_chunk].x1, bbs[bbs_pos_in_chunk].y1, bbs[bbs_pos_in_chunk].x2, bbs[bbs_pos_in_chunk].y2))
                #print('----\tx1:{:.2f}\ty1:{:.2f}\tx2:{:.2f}\ty2:{:.2f}\t'.format(scaled_bb.x1, scaled_bb.y1, scaled_bb.x2, scaled_bb.y2))
            bbs_pos_in_chunk += 1
        
        # This loop uses its inner loop to compare all gt_bbs with its current bb until finding one,
        # then the inner loop breaks to get back to this loop and work on the next bb.
        bbs_pos_in_frame = 0
        true_positive, false_positive, false_negative = 0, 0, 0
        while bbs_pos_in_frame < len(bbs_per_frame):
            bb = bbs_per_frame[bbs_pos_in_frame]
            gt_bbs_pos_in_frame = 0
            while gt_bbs_pos_in_frame < len(gt_bbs_per_frame):
                gt_bb = gt_bbs_per_frame[gt_bbs_pos_in_frame]
                # Check if the two bounding boxes correspond to each other
                if gt_bb.object != bb.object or not is_overlapping(gt_bb, bb):
                    gt_bbs_pos_in_frame += 1
                else:
                    # Reference: https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
                    
                    # Compute intersection between the two bbs, i.e., the common box part between the two bbs
                    x1_inter, y1_inter = max(gt_bb.x1, bb.x1), max(gt_bb.y1, bb.y1)
                    x2_inter, y2_inter = min(gt_bb.x2, bb.x2), min(gt_bb.y2, bb.y2)
                    width_inter, height_inter = x2_inter - x1_inter, y2_inter - y1_inter
                    area_inter = width_inter * height_inter
                    
                    # Compute union area of the two bbs, i.e., the sum of their areas minus the area of intersection
                    gt_bb_area = (gt_bb.x2 - gt_bb.x1) * (gt_bb.y2 - gt_bb.y1)
                    bb_area = (bb.x2 - bb.x1) * (bb.y2 - bb.y1)
                    area_union = gt_bb_area + bb_area - area_inter
                    
                    IoU = area_inter / area_union
                    # Reference: https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b
                    if IoU >= 0.5:
                        true_positive += 1
                    else:
                        false_positive += 1
                    bbs_per_frame.pop(bbs_pos_in_frame)
                    gt_bbs_per_frame.pop(gt_bbs_pos_in_frame)
                    break
            if len(bbs_per_frame) > 0 and gt_bbs_pos_in_frame == len(gt_bbs_per_frame):
                bbs_per_frame.pop(bbs_pos_in_frame)
        
        false_negative += len(gt_bbs_per_frame)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = (2 * precision * recall) / (precision + recall)
        f1_scores_for_frame.append(f1_score)
        
        with open(results_file_path, 'a') as res_file:
            res_file.write('{}:\t{:.2f}\n'.format(image_name, f1_score))
            
    with open(results_file_path, 'a') as res_file:
        res_file.write('f1 score of chunk (average):\t{:.2f}'.format(np.average(f1_scores_for_frame)))
    return np.average(f1_scores_for_frame) # f1 score of chunk, i.e., average f1 score of all frames in this chunk

def f1_scores(data_dir, content, num_chunks):
    parent_path = os.path.join(data_dir, 'experiment-outputs')
    
    orig_dir = os.path.join(parent_path, 'accuracy_results', '240p_f1_scores')
    os.makedirs(orig_dir, exist_ok=True)
    
    nemo_sr_dir = os.path.join(parent_path, 'accuracy_results', '1080p_nemo_sr_f1_scores')
    os.makedirs(nemo_sr_dir, exist_ok=True)

    per_bb_sr_dir = os.path.join(parent_path, 'accuracy_results', '1080p_per_bb_sr_f1_scores')
    os.makedirs(per_bb_sr_dir, exist_ok=True)
    
    per_small_bb_sr_dir = os.path.join(parent_path, 'accuracy_results', '1080p_per_small_bb_sr_f1_scores')
    os.makedirs(per_small_bb_sr_dir, exist_ok=True)
    
    orig_file_name = '240p_bbs'
    nemo_sr_file_name = '1080p_nemo_sr_bbs'
    per_bb_sr_file_name = '1080p_per_bb_sr_bbs'
    per_frame_sr_file_name = '1080p_per_frame_sr_bbs'
    
    orig_f1_score_per_chk, nemo_sr_f1_score_per_chk, per_bb_sr_f1_score_per_chk, per_small_bb_sr_f1_score_per_chk = [], [], [], []
    for chunk_idx in range(0, num_chunks):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        gt_bbs_file_path = os.path.join(parent_path, '2-1080p---per-frame-sr', postfix, per_frame_sr_file_name)
        
        # original 240p case
        bbs_file_path = os.path.join(parent_path, '1-240p', postfix, orig_file_name)
        res = compute_f1_scores_for_chunk(gt_bbs_file_path, bbs_file_path, os.path.join(orig_dir, postfix + '.txt'), False, 426, 240, 1920, 1080)
        orig_f1_score_per_chk.append(res)
               
        # 1080p nemo sr case
        bbs_file_path = os.path.join(parent_path, '4-1080p---nemo-sr---any-ap---no-max-aps', postfix, nemo_sr_file_name)
        res = compute_f1_scores_for_chunk(gt_bbs_file_path, bbs_file_path, os.path.join(nemo_sr_dir, postfix + '.txt'))
        nemo_sr_f1_score_per_chk.append(res)
        
        # 1080p per bb sr case
        bbs_file_path = os.path.join(parent_path, '5-1080p---per-bb-sr---any-ap---max-8-aps', postfix, per_bb_sr_file_name)
        res = compute_f1_scores_for_chunk(gt_bbs_file_path, bbs_file_path, os.path.join(per_bb_sr_dir, postfix + '.txt'))
        per_bb_sr_f1_score_per_chk.append(res)
        
        # 1080p per small bb sr case
        bbs_file_path = os.path.join(parent_path, '7-1080p---per-small-bb-sr---any-ap---max-8-aps', postfix, per_bb_sr_file_name)
        res = compute_f1_scores_for_chunk(gt_bbs_file_path, bbs_file_path, os.path.join(per_small_bb_sr_dir, postfix + '.txt'))
        per_small_bb_sr_f1_score_per_chk.append(res)
        
    with open(os.path.join(parent_path, 'accuracy_results', 'summary-f1-scores-of-video.txt'), 'a') as res_file:
        res_file.write('240p:\t\t\t{:.2f}\n'.format(np.average(orig_f1_score_per_chk)))
        res_file.write('1080p nemo sr:\t\t{:.2f}\n'.format(np.average(nemo_sr_f1_score_per_chk)))
        res_file.write('1080p per bb sr:\t{:.2f}\n'.format(np.average(per_bb_sr_f1_score_per_chk)))
        res_file.write('1080p per small bb sr:\t{:.2f}\n'.format(np.average(per_small_bb_sr_f1_score_per_chk)))
        
f1_scores('/home/mai/nemo/data/nemo-data-traffic/traffic', 'traffic', 25)
