from dds_utils import (Results, read_results_dict, evaluate, cleanup, Region,
                       compute_regions_size, merge_boxes_in_results, extract_images_from_video,calc_iou,filter_bbox_group)
import os
import logging
import torch
import cv2 as cv
from backend.object_detector import Detector
import time

# Mai, final task
#import argparse
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--images_direc', type=str, required=True)
#    args = parser.parse_args()
# end

def detect_objects(images_direc, det_out_file):
    final_results = Results()
    detector=Detector()
    #images_direc = args.images_direc
    #images_direc='/home/ubuntu/VideoAnalytics/mi_dds_sr/Faster-RCNN/video_test/src/'
    fnames = sorted(os.listdir(images_direc))
    for fname in fnames:
        t1=time.time()
        if "png" not in fname:
            continue
        # fid = int(fname.split(".")[0]) Mai, final task, commented
        fid = fname.split(".")[0] # Mai, final task, added since we have _'s in the images of some chunks
        
        image = None
        #if fid>10:
        #    break
        image_path = os.path.join(images_direc, fname)
        image = cv.imread(image_path)
        #image=cv.resize(image, (480, 270))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        detection_results = detector.infer(image)

        # (label, conf, box_tuple)
        # print(detection_results)
        frame_with_no_results = True
        for label, conf, (x, y, w, h) in detection_results:
            r = Region(fid, x, y, w, h, conf, label, 1, origin="mpeg")
            #if conf < 0.5: # Mai, final task, added to filter out less confident BBs
            #    continue
            final_results.append(r)
            frame_with_no_results = False
        t2=time.time()
        #print(t2-t1)
        if frame_with_no_results:
            final_results.append(
                Region(fid, 0, 0, 0, 0, 0.1, "no obj", 1))
        # Mai, final task, commented since now, fid is str not int
        #if (fid+1)%100==0:
            #print('detect fid',fid)
    final_results.write(det_out_file)
