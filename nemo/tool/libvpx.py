import math
import os
import struct
import copy
import subprocess
import shlex
import time
import gc

from PIL import Image as im # Mai, final task, added

import tensorflow as tf

from nemo.tool.video import get_video_profile
from nemo.dnn.dataset import single_raw_dataset, single_raw_dataset_with_name
from nemo.dnn.utility import resolve_bilinear # Mai, added for task 3

class Frame():
    def __init__(self, video_index, super_index):
        self.video_index = video_index
        self.super_index= super_index

    @property
    def name(self):
        return '{}.{}'.format(self.video_index, self.super_index)

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.video_index == other.video_index and self.super_index == other.super_index:
                return True
            else:
                return False
        else:
            return False

class AnchorPointSet():
    def __init__(self, frames, anchor_point_set, save_dir, name):
        assert (frames is None or anchor_point_set is None)

        if frames is not None:
            self.frames = frames
            self.anchor_points = []
            self.estimated_quality = None
            self.measured_quality = None

        if anchor_point_set is not None:
            self.frames = copy.deepcopy(anchor_point_set.frames)
            self.anchor_points = copy.deepcopy(anchor_point_set.anchor_points)
            self.estimated_quality = copy.deepcopy(anchor_point_set.estimated_quality)
            self.measured_quality = copy.deepcopy(anchor_point_set.measured_quality)

        self.save_dir = save_dir
        self.name = name

    @classmethod
    def create(cls, frames, save_dir, name):
        return cls(frames, None, save_dir, name)

    @classmethod
    def load(cls, anchor_point_set, save_dir, name):
        return cls(None, anchor_point_set, save_dir, name)

    @property
    def path(self):
        return os.path.join(self.save_dir, self.name)

    def add_anchor_point(self, frame, quality=None):
        self.anchor_points.append(frame)
        self.quality = quality

    def get_num_anchor_points(self):
        return len(self.anchor_points)
        
    def get_cache_profile_name(self):
        return self.name

    def set_cache_profile_name(self, name):
        self.name = name

    def get_estimated_quality(self):
        return self.estimated_quality

    def get_measured_quality(self, quality):
        return self.measured_quality

    def set_estimated_quality(self, quality):
        self.estimated_quality = quality

    def set_measured_quality(self, quality):
        self.measured_quality = quality

    def save_cache_profile(self):
        path = os.path.join(self.save_dir, '{}.profile'.format(self.name))

        num_remained_bits = 8 - (len(self.frames) % 8)
        num_remained_bits = num_remained_bits % 8

        with open(path, "wb") as f:
            f.write(struct.pack("=I", num_remained_bits))

            byte_value = 0
            for i, frame in enumerate(self.frames):
                if frame in self.anchor_points:
                    byte_value += 1 << (i % 8)

                if i % 8 == 7:
                    f.write(struct.pack("=B", byte_value))
                    byte_value = 0

            if len(self.frames) % 8 != 0:
                f.write(struct.pack("=B", byte_value))

    def remove_cache_profile(self):
        cache_profile_path = os.path.join(self.save_dir, '{}.profile'.format(self.name))
        if os.path.exists(cache_profile_path):
            os.remove(cache_profile_path)

    def __lt__(self, other):
        return self.count_anchor_points() < other.count_anchor_points()

def load_frame_index(dataset_dir, video_name, postfix=None):
    frames = []
    if postfix is None:
        log_path = os.path.join(dataset_dir, 'log', video_name, 'metadata.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', video_name, postfix, 'metadata.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            current_video_frame = int(line.split('\t')[0])
            current_super_frame = int(line.split('\t')[1])
            frames.append(Frame(current_video_frame, current_super_frame))

    return frames

def save_rgb_frame(vpxdec_path, dataset_dir, video_name, output_width=None, output_height=None, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)
    
    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={}  \
        --input-video-name={} --threads={} --save-rgbframe --save-metadata'.format(vpxdec_path, dataset_dir, video_name, get_num_threads(video_profile['height']))
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if output_width is not None:
        command += ' --output-width={}'.format(output_width)
    if output_height is not None:
        command += ' --output-height={}'.format(output_height)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_yuv_frame(vpxdec_path, dataset_dir, video_name, output_width=None, output_height=None, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)
    
    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={}  \
        --input-video-name={} --threads={} --save-yuvframe --save-metadata'.format(vpxdec_path, dataset_dir, video_name, get_num_threads(video_profile['height']))
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if output_width is not None:
        command += ' --output-width={}'.format(output_width)
    if output_height is not None:
        command += ' --output-height={}'.format(output_height)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_metadata(vpxdec_path, dataset_dir, video_name, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --content={} \
        --input-video-name={} --threads={} --save-rgbframe'.format(vpxdec_path, dataset_dir, content, video_name, get_num_threads(video_profile['height']))
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def setup_sr_frame_task3(vpxdec_path, dataset_dir, video_name, model, postfix=None, is_dnn=True, scale=4): # added is_dnn and scale, Mai
    if postfix is None:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name)
    else:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name, postfix)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    single_raw_ds = single_raw_dataset_with_name(lr_image_dir, video_profile['width'], video_profile['height'], 3, exp='.raw')
    for idx, img in enumerate(single_raw_ds):
        lr = img[0] # tensor size:  lr:(1, 240, 426, 3) sr:(1, 960, 1704, 3) Mai
        lr = tf.cast(lr, tf.float32)
        # task 3, Mai
        if (is_dnn):
            sr = model(lr)
        else:
            #sr = lr # trial
            # Apply sr dnn to the upper left quarter of the tensor and bilinear interpolation to the rest Mai
            lr_rows = lr.shape[1]
            lr_cols = lr.shape[2]
            
            # Attempt 1: sr dnn for upper left quarter
            #sr1 = model(lr[:, :60, :106, :])
            
            #bi1 = resolve_bilinear(lr[:, :60, 106:, :], lr[:, :60, 106:, :].shape[1] * scale, lr[:, :60, 106:, :].shape[2] * scale)
            #bi1 = tf.cast(bi1, tf.float32) # for tf syntax reasons
            
            #bi2 = resolve_bilinear(lr[:, 60:, :, :], lr[:, 60:, :, :].shape[1] * scale, lr[:, 60:, :, :].shape[2] * scale)
            #bi2 = tf.cast(bi2, tf.float32)
            
            #sr = tf.concat([sr1, bi1], 2) # 0:batch 1:rows 2:columns 3:channels
            #sr = tf.concat([sr, bi2], 1)
            
            
            # Attempt 2: sr dnn for center half (central box), i.e. half the number of pixels in the center of the frame
            hor_beg = int(lr_rows/2 - lr_rows/4)
            hor_end = int(lr_rows/2 + lr_rows/4)
            ver_beg = int(lr_cols/2 - lr_cols/4)
            ver_end = int(lr_cols/2 + lr_cols/4)
            
            sr1 = model(lr[:, hor_beg:hor_end, ver_beg:ver_end, :])
            
            bi1 = resolve_bilinear(lr[:, :hor_beg, :, :], lr[:, :hor_beg, :, :].shape[1] * scale, lr[:, :hor_beg, :, :].shape[2] * scale) # above central box
            bi1 = tf.cast(bi1, tf.float32) # for tf syntax reasons
            
            bi2 = resolve_bilinear(lr[:, hor_end:, :, :], lr[:, hor_end:, :, :].shape[1] * scale, lr[:, hor_end:, :, :].shape[2] * scale) # below central box
            bi2 = tf.cast(bi2, tf.float32)
            
            bi3 = resolve_bilinear(lr[:, hor_beg:hor_end, ver_end:, :], lr[:, hor_beg:hor_end, ver_end:, :].shape[1] * scale,
                                   lr[:, hor_beg:hor_end, ver_end:, :].shape[2] * scale) # at the right of central box
            bi3 = tf.cast(bi3, tf.float32)
            
            bi4 = resolve_bilinear(lr[:, hor_beg:hor_end, :ver_beg, :], lr[:, hor_beg:hor_end, :ver_beg, :].shape[1] * scale,
                                   lr[:, hor_beg:hor_end, :ver_beg, :].shape[2] * scale) # at the left of central box
            bi4 = tf.cast(bi4, tf.float32)

            sr = tf.concat([sr1, bi3], 2) # 0:batch 1:rows 2:columns 3:channels
            sr = tf.concat([bi4, sr], 2)
            sr = tf.concat([bi1, sr], 1)
            sr = tf.concat([sr, bi2], 1)
	
        sr = tf.clip_by_value(sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)

        sr_image = tf.squeeze(sr).numpy()
        name = os.path.basename(img[1].numpy()[0].decode())
        sr_image.tofile(os.path.join(sr_image_dir, name))

        #validate
        #sr_image = tf.image.encode_png(tf.squeeze(sr))
        #tf.io.write_file(os.path.join(sr_image_dir, '{0:04d}.png'.format(idx+1)), sr_image)

######################################## Mai, final task ########################################
def get_png_of_raw(input_dir, width, height, output_dir=None):
    single_raw_ds = single_raw_dataset_with_name(input_dir, width, height, 3, exp='.raw')

    if output_dir is None:
        output_dir = os.path.join(input_dir, 'pngs')
    os.makedirs(output_dir, exist_ok=True)

    for idx, img in enumerate(single_raw_ds):
        sr = img[0]
        sr = tf.cast(sr, tf.float32)
        
        sr = tf.clip_by_value(sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)
        sr = tf.squeeze(sr).numpy()
        
        name = os.path.basename(img[1].numpy()[0].decode())
        name = name.replace("raw", "png")
        
        data = im.fromarray(sr)
        data.save(output_dir + '/' + name)
        
class bounding_box:
    def __init__(self, image_name, x1, y1, x2, y2, object, score):
        self.image_name = image_name
        
        self.x1 = round(float(x1)) # col 1
        self.y1 = round(float(y1)) # row 1
        self.x2 = round(float(x2)) # col 2
        self.y2 = round(float(y2)) # row 2
        
        if self.x1 == 0:
            self.x1 = 1
            self.x2 += 1
        if self.y1 == 0:
            self.y1 = 1
            self.y2 += 1
            
        self.object = object
        self.score = float(score)
        
def parse_obj_det_res(filepath):
    with open(filepath, 'r') as f:
        bounding_boxes = []
        lines = f.readlines()
    
        for line in lines:
            line = line.split(',')
            bounding_boxes.append(bounding_box(line[0], line[1], line[2], line[3], line[4], line[5], line[6]))
        f.close()
    return bounding_boxes

# I leave this function here as a demonstration of why the main setup_sr_frame_bbox can't be optimized to work with batches. Use this function to see the error in detail. BBs are of different sizes.
def setup_sr_frame_bbox_batch_over_bbs(dataset_dir, video_name, model, postfix=None, scale=4, sr_small_bbs_only=False):
    if postfix is None:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name)
    else:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name, postfix)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    single_raw_ds = single_raw_dataset_with_name(lr_image_dir, video_profile['width'], video_profile['height'], 3, exp='.raw')
    
    #total_sr_time = 0
    for idx, img in enumerate(single_raw_ds):
        name = os.path.basename(img[1].numpy()[0].decode())
        image_name = name.split('.')[0] # to remove extension .raw
                
        lr = img[0] # tensor size:  lr:(1, 240, 426, 3) sr:(1, 960, 1704, 3) Mai
        lr = tf.cast(lr, tf.float32)    
        
        if '_' in image_name:
            continue
        obj_det_res_path = os.path.join(lr_image_dir, 'det_results', image_name)    
        
        with open(obj_det_res_path, 'r') as f:
            bbs, bbs_tensors = [], []
            lines = f.readlines()
    
            for line in lines:
                line = line.split(',')
                bb = bounding_box(line[0], line[1], line[2], line[3], line[4], line[5], line[6])
                bbs.append(bbs)
                bbs_tensors.append(lr[:, bb.y1-1:bb.y2, bb.x1-1:bb.x2, :])
            f.close()
        
        bbs_tensors_ds = tf.data.experimental.from_list(bbs_tensors).batch(500)
        for idx2, bbs in enumerate(bbs_tensors_ds):
            print(bbs)
            sr = model(bbs)
        bi = resolve_bilinear(lr, lr.shape[1] * scale, lr.shape[2] * scale)
        bi = tf.cast(bi, tf.float32)
        bi_sr = bi
        
        for bbs_idx, bbs in enumerate(bbs):
            r1 = (bb.y1-1) * scale
            r2 = r1 - 1 + ((bb.y2 - bb.y1 + 1) * scale)
            c1 = (bb.x1-1) * scale
            c2 = c1 - 1 + ((bb.x2 - bb.x1 + 1) * scale)
            
            # bi[:, r1:r2+1, c1:c2+1, :] = sr # Wrong syntax, just keep it since it explains what the following lines do.
            bi_sr = tf.concat([bi[:, :r1, c1:c2+1, :], sr[bbs_idx]], axis=1)
            bi_sr = tf.concat([bi_sr[:, :, :, :], bi[:, r2+1:, c1:c2+1, :]], axis=1)
            bi_sr = tf.concat([bi[:, :, :c1, :], bi_sr], axis=2)
            bi_sr = tf.concat([bi_sr[:, :, :, :], bi[:, :, c2+1:, :]], axis=2)
            bi = bi_sr
                
        sr = tf.clip_by_value(bi_sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)
        
        sr_image = tf.squeeze(sr).numpy()
        sr_image.tofile(os.path.join(sr_image_dir, name))
    return 0
    
def setup_sr_frame_bbox(dataset_dir, video_name, model, postfix=None, scale=4, sr_small_bbs_only=False):
    if postfix is None:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name)
    else:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name, postfix)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    single_raw_ds = single_raw_dataset_with_name(lr_image_dir, video_profile['width'], video_profile['height'], 3, exp='.raw')
    obj_det_res_path = os.path.join(lr_image_dir, 'det_results')
    bbs = parse_obj_det_res(obj_det_res_path)
    total_sr_time = 0
    bb_i = 0  # bounding box index
    for idx, img in enumerate(single_raw_ds):
        name = os.path.basename(img[1].numpy()[0].decode())
        image_name = name.split('.')[0] # to remove extension .raw
        
        lr = img[0] # tensor size:  lr:(1, 240, 426, 3) sr:(1, 960, 1704, 3) Mai
        lr = tf.cast(lr, tf.float32)
        
        start_time = time.time()
        bi = resolve_bilinear(lr, lr.shape[1] * scale, lr.shape[2] * scale)
        #bi = tf.zeros([1, lr.shape[1] * scale, lr.shape[2] * scale, 3], tf.float32) # For tracing
        bi = tf.cast(bi, tf.float32)
        #bi_sr = bi
        total_sr_time += (time.time() - start_time)
        
        # Optimized search using bb_i and breaking upon reaching a different image
        #print("Length bbs: ", len(bbs), "bbs[bb_i].image_name: ", bbs[bb_i].image_name, "image_name: ", image_name)
        while bb_i < len(bbs) and bbs[bb_i].image_name == image_name:
            #if bbs[bb_i].score < 0.5: # Mai, final task, filter out bbs of low confidence score
            #    bb_i += 1
            #    continue
            # optional task, sr only small bbs, e.g. < 15x15
            '''
            if sr_small_bbs_only:
                area = (bbs[bb_i].x2 - bbs[bb_i].x1) * (bbs[bb_i].y2 - bbs[bb_i].y1)
                if area >= 20**2:
                    bb_i += 1
                    continue
            '''        
            start_time = time.time()
            sr = model(lr[:, bbs[bb_i].y1-1:bbs[bb_i].y2, bbs[bb_i].x1-1:bbs[bb_i].x2, :])
            total_sr_time += (time.time() - start_time)
            
            r1 = (bbs[bb_i].y1-1) * scale
            r2 = r1 - 1 + ((bbs[bb_i].y2 - bbs[bb_i].y1 + 1) * scale)
            c1 = (bbs[bb_i].x1-1) * scale
            c2 = c1 - 1 + ((bbs[bb_i].x2 - bbs[bb_i].x1 + 1) * scale)
            
            # bi[:, r1:r2+1, c1:c2+1, :] = sr # Wrong syntax, just keep it since it explains what the following lines do.
            #bi_sr = tf.concat([bi[:, :r1, c1:c2+1, :], sr], axis=1)
            #bi_sr = tf.concat([bi_sr[:, :, :, :], bi[:, r2+1:, c1:c2+1, :]], axis=1)
            #bi_sr = tf.concat([bi[:, :, :c1, :], bi_sr], axis=2)
            #bi_sr = tf.concat([bi_sr[:, :, :, :], bi[:, :, c2+1:, :]], axis=2)
            bb_i += 1
            bi = tf.Variable(bi)
            bi[:, r1:r2+1, c1:c2+1, :].assign(sr)
            #bi = bi_sr
            
            # For tracing
            #with open("/home/mai/nemo/trace_file.txt", "a") as trace_f:
            #    trace_f.write("Bounding box index: {}, image name: {}, scale: {}\n".format(bb_i, image_name, scale))
            #    trace_f.write("\n1st channel:\n")
            #    trace_f.write('r1={}, r2={}, c1={}, c2={}\n{}\n'.format(r1, r2, c1, c2, bi_sr[0, r1-1:r2+1+1, c1-1:c2+1+1, 0]))
            #    trace_f.write("\2nd channel:\n")
            #    trace_f.write('r1={}, r2={}, c1={}, c2={}\n{}\n'.format(r1, r2, c1, c2, bi_sr[0, r1:r2+1, c1:c2+1, 1]))
            #    trace_f.write("\3rd channel:\n")
            #    trace_f.write('r1={}, r2={}, c1={}, c2={}\n{}\n'.format(r1, r2, c1, c2, bi_sr[0, r1:r2+1, c1:c2+1, 2]))
            #break
        #print("BB index: ", bb_i)
        sr = tf.clip_by_value(bi, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)
        
        sr_image = tf.squeeze(sr).numpy()
        sr_image.tofile(os.path.join(sr_image_dir, name))
    return total_sr_time

def setup_sr_frame_dnn(dataset_dir, video_name, model, postfix=None, scale=4):
    if postfix is None:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name)
    else:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name, postfix)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    single_raw_ds = single_raw_dataset_with_name(lr_image_dir, video_profile['width'], video_profile['height'], 3, exp='.raw')
    for idx, img in enumerate(single_raw_ds):
        lr = img[0]
        lr = tf.cast(lr, tf.float32)
        
        # Mai, added the following lines to compare apple to apple, since I have to apply bilinear interpolation with bounding boxes too.
        # This way, the difference in latencies would be accurate.
        bi_sr = resolve_bilinear(lr, lr.shape[1] * scale, lr.shape[2] * scale)
        bi_sr = tf.cast(bi_sr, tf.float32)
        
        sr = model(lr)
        
        sr = tf.clip_by_value(sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)

        sr_image = tf.squeeze(sr).numpy()
        name = os.path.basename(img[1].numpy()[0].decode())
        sr_image.tofile(os.path.join(sr_image_dir, name))

def setup_sr_frame_nemo(vpxdec_path, dataset_dir, video_name, model, postfix=None, is_dnn=True, scale=4, sr_small_bbs_only=False):
    if (is_dnn):
        setup_sr_frame_dnn(dataset_dir, video_name, model, postfix, scale)
        return 0
    else:
        return setup_sr_frame_bbox(dataset_dir, video_name, model, postfix, scale, sr_small_bbs_only) # sr_small_bbs_only is for the optional task
######################################## Mai, final task ########################################

def setup_sr_frame(vpxdec_path, dataset_dir, video_name, model, postfix=None):
    if postfix is None:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name)
    else:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name, postfix)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    single_raw_ds = single_raw_dataset_with_name(lr_image_dir, video_profile['width'], video_profile['height'], 3, exp='.raw')
    for idx, img in enumerate(single_raw_ds):
        lr = img[0]
        lr = tf.cast(lr, tf.float32)
        sr = model(lr)

        sr = tf.clip_by_value(sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)

        sr_image = tf.squeeze(sr).numpy()
        name = os.path.basename(img[1].numpy()[0].decode())
        sr_image.tofile(os.path.join(sr_image_dir, name))

        #validate
        #sr_image = tf.image.encode_png(tf.squeeze(sr))
        #tf.io.write_file(os.path.join(sr_image_dir, '{0:04d}.png'.format(idx+1)), sr_image)

def bilinear_quality(vpxdec_path, dataset_dir, input_video_name, reference_video_name,
                               output_width, output_height, skip=None, limit=None, postfix=None):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, postfix, 'quality.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_video_profile = get_video_profile(input_video_path)

    if not os.path.exists(log_path):
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} \
            --output-width={} --output-height={} --save-quality --save-metadata --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, \
                                                                        output_width, output_height, get_num_threads(input_video_profile['height']))
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    quality = []

    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

def offline_dnn_quality(vpxdec_path, dataset_dir, input_video_name, reference_video_name,  \
                                model_name, output_width, output_height, skip=None, limit=None, postfix=None, \
                                save_imgs=None, save_imgs_dir=None):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, 'quality.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = get_video_profile(input_video_path)['height']
    scale = output_height // input_resolution

    if not os.path.exists(log_path):
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} \
        --dnn-scale={} --dnn-name={} --output-width={} --output-height={} --decode-mode=decode_sr --dnn-mode=offline_dnn --save-quality --save-metadata \
            --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, scale, model_name, output_width, output_height, get_num_threads(input_resolution))
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        if save_imgs is not None:
            command += ' --save-imgs={}'.format(save_imgs)
            command += ' --save-imgs-dir={}'.format(save_imgs_dir)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    quality = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

def save_cache_frame(vpxdec_path, dataset_dir, input_video_name, reference_video_name,  \
                                model_name, cache_profile_file, resolution, skip=None, limit=None, postfix=None):
    #log file
    log_dir = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_file))
    if postfix is not None:
        log_dir = os.path.join(log_dir, postfix)
    log_path = os.path.join(log_dir, 'quality.txt')

    #run sr-integrated decoder
    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} \
    --input-video-name={} --reference-video-name={} --decode-mode=decode_cache --dnn-mode=offline_dnn --cache-mode=profile_cache \
    --save-quality --save-frame --save-metadata --dnn-name={} --cache-profile-name={} --resolution={}'.format(vpxdec_path, dataset_dir, input_video_name, \
                                                    reference_video_name, model_name, cache_profile_file, resolution)
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    #subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL)

def offline_cache_metadata(vpxdec_path, dataset_dir, input_video_name,  \
                                model_name, cache_profile_name, output_width, output_height, skip=None, limit=None, postfix=None):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'metadata.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'metadata.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = get_video_profile(input_video_path)['height']
    scale = output_height // input_resolution

    if not os.path.exists(log_path):
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} \
        --input-video-name={} --decode-mode=decode_cache --dnn-mode=offline_dnn --cache-mode=profile_cache \
        --output-width={} --output-height={} --save-metadata --dnn-name={} --dnn-scale={} --cache-profile-name={}'.format(vpxdec_path, \
                            dataset_dir, input_video_name, output_width, output_height, model_name, scale, cache_profile_name)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def offline_cache_quality(vpxdec_path, dataset_dir, input_video_name, reference_video_name,  \
                                model_name, cache_profile_name, output_width, output_height, skip=None, limit=None, postfix=None \
                                , save_imgs=None, save_imgs_dir=None):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'quality.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = get_video_profile(input_video_path)['height']
    scale = output_height // input_resolution

    if not os.path.exists(log_path):
        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} \
        --input-video-name={} --reference-video-name={} --decode-mode=decode_cache --dnn-mode=offline_dnn --cache-mode=profile_cache \
        --output-width={} --output-height={} --save-quality --save-metadata --dnn-name={} --dnn-scale={} --cache-profile-name={}'.format(vpxdec_path, \
                            dataset_dir, input_video_name, reference_video_name, output_width, output_height, model_name, scale, cache_profile_name)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        if save_imgs is not None:
            command += ' --save-imgs={}'.format(save_imgs)
            command += ' --save-imgs-dir={}'.format(save_imgs_dir)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    quality = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))

    return quality

def offline_cache_quality_mt_v1(q0, q1, vpxdec_path, dataset_dir, input_video_name, reference_video_name, model_name, output_width, output_height):
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = get_video_profile(input_video_path)['height']
    scale = output_height // input_resolution

    while True:
        item = q0.get()
        if item == 'end':
            return
        else:
            start_time = time.time()
            anchor_point_set = item[0]
            skip = item[1]
            limit = item[2]
            postfix = item[3]

            #log file
            if postfix is not None:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(anchor_point_set.get_cache_profile_name()), 'quality.txt')
            else:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(anchor_point_set.get_cache_profile_name()), 'quality.txt')

            #run sr-integrated decoder
            anchor_point_set.save_cache_profile()
            command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} --decode-mode=decode_cache \
            --dnn-mode=offline_dnn --cache-mode=profile_cache --output-width={} --output-height={} --save-quality --save-metadata --dnn-name={} --dnn-scale={} \
            --cache-profile-name={} --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, output_width, output_height, \
                                                         model_name, scale, anchor_point_set.get_cache_profile_name(), get_num_threads(input_resolution))
            if skip is not None:
                command += ' --skip={}'.format(skip)
            if limit is not None:
                command += ' --limit={}'.format(limit)
            if postfix is not None:
                command += ' --postfix={}'.format(postfix)
            subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            #result = subprocess.check_output(shlex.split(command)).decode('utf-8')
            #result = result.split('\n')
            anchor_point_set.remove_cache_profile()

            #load quality from a log file
            quality = []
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    quality.append(float(line.split('\t')[1]))
            end_time = time.time()

            q1.put(quality)

def offline_cache_quality_mt(q0, q1, vpxdec_path, dataset_dir, input_video_name, reference_video_name, model_name, output_width, output_height):
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = get_video_profile(input_video_path)['height']
    scale = output_height // input_resolution

    while True:
        item = q0.get()
        if item == 'end':
            return
        else:
            start_time = time.time()
            cache_profile_name = item[0]
            skip = item[1]
            limit = item[2]
            postfix = item[3]
            idx = item[4]

            #log file
            if postfix is not None:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'quality.txt')
            else:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'quality.txt')

            #run sr-integrated decoder
            if not os.path.exists(log_path):
                command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} --decode-mode=decode_cache \
                --dnn-mode=offline_dnn --cache-mode=profile_cache --output-width={} --output-height={} --save-quality --save-metadata --dnn-name={} --dnn-scale={} \
                --cache-profile-name={} --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, output_width, output_height, \
                                                             model_name, scale, cache_profile_name, get_num_threads(input_resolution))
                if skip is not None:
                    command += ' --skip={}'.format(skip)
                if limit is not None:
                    command += ' --limit={}'.format(limit)
                if postfix is not None:
                    command += ' --postfix={}'.format(postfix)
                subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                #result = subprocess.check_output(shlex.split(command)).decode('utf-8')
                #result = result.split('\n')
                
            #load quality from a log file
            quality = []
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:

                    line = line.strip()
                    quality.append(float(line.split('\t')[1]))
            end_time = time.time()

            q1.put((idx, quality))

#ref: https://developers.google.com/media/vp9/settings/vod
def get_num_threads(resolution):
    tile_size = 256
    if resolution >= tile_size:
        num_tiles = resolution // tile_size
        log_num_tiles = math.floor(math.log(num_tiles, 2))
        num_threads = (2**log_num_tiles) * 2
    else:
        num_threads = 2
    return num_threads

def count_mac_for_cache(width, height, channel):
    return width * height * channel * 8

if __name__ == '__main__':
    frame_list = [Frame(0,1)]
    frame1 = Frame(0,1)
    print(frame1 == frame_list[0])
