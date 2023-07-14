import os
import sys
import argparse
import shlex
import math
import time
import multiprocessing as mp
import shutil
import random
import itertools

from os import listdir # Mai, final task, added

import numpy as np
import tensorflow as tf
#tf.enable_eager_execution() # My Fix

from nemo.tool.video import get_video_profile
import nemo.tool.libvpx as libvpx
from nemo.tool.mac import count_mac_for_dnn, count_mac_for_cache
import nemo.object_detector.Backend.detect_fn as objdet # Mai, final task, added

class AnchorPointSelector():
    def __init__(self, model, vpxdec_path, dataset_dir, lr_video_name, hr_video_name, gop, output_width, output_height, \
                 quality_margin, num_decoders, scale=None, content=None): # added scale and content, Mai
        self.model = model
        self.vpxdec_path = vpxdec_path
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.gop = gop
        self.output_width = output_width
        self.output_height = output_height
        self.quality_margin = quality_margin
        self.num_decoders = num_decoders
        self.scale = scale # added, Mai
        self.content = content # added, Mai

    #select an anchor point that maiximizes the quality gain
    def _select_anchor_point(self, current_anchor_point_set, anchor_point_candidates):
        max_estimated_quality = None
        max_avg_estimated_quality = None
        idx = None

        for i, new_anchor_point in enumerate(anchor_point_candidates):
            estimated_quality = self._estimate_quality(current_anchor_point_set, new_anchor_point)
            avg_estimated_quality = np.average(estimated_quality)
            if max_avg_estimated_quality is None or avg_estimated_quality > max_avg_estimated_quality:
                max_avg_estimated_quality = avg_estimated_quality
                max_estimated_quality = estimated_quality
                idx = i

        return idx, max_estimated_quality

    #estimate the quality of an anchor point set
    def _estimate_quality(self, current_anchor_point_set, new_anchor_point):
        if current_anchor_point_set is not None:
            return np.maximum(current_anchor_point_set.estimated_quality, new_anchor_point.measured_quality)
        else:
            return new_anchor_point.measured_quality

    #select an anchor point set using the NEMO algorithm
    def _select_anchor_point_set_nemo(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)
        algorithm_type = 'nemo_{}'.format(self.quality_margin)
        
        ###########step 1: analyze anchor points##########
        #calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = get_video_profile(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames
        
        #save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        start_time_dnn = time.time() # Mai, final task, added
        libvpx.setup_sr_frame_nemo(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.model, postfix, is_dnn=True, scale=self.scale) # added is_dnn and scale, Mai, final task, changed setup_sr_frame to setup_sr_frame_nemo
        end_time_dnn = time.time() # Mai, final task, added
        
        end_time = time.time()
        print('{} video chunk: (Step1-profile bilinear, dnn quality) {}sec'.format(chunk_idx, end_time - start_time))

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                   num_skipped_frames, num_decoded_frames, postfix)

        save_imgs_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix, 'per_frame_sr') # Mai, final task, added
        quality_dnn = libvpx.offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix, \
                                                save_imgs=1, save_imgs_dir=save_imgs_dir) # Mai, final task, added in response to my changes in the sr-integrated codec to save .raw results
                                                
        #### Mai, final task, beg ####
        dim_file_path = os.path.join(save_imgs_dir, 'width_height.txt')
        with open(dim_file_path, 'r') as dim_file:
            line = dim_file.readline()
            output_width, output_height = int(line.split('\t')[0]), int(line.split('\t')[1])
        pngs_parent_dir = os.path.join(self.dataset_dir, 'detection_results', self.content, postfix)
        os.makedirs(pngs_parent_dir, exist_ok=True)
        pngs_dir = os.path.join(pngs_parent_dir, str(output_height) + 'p_per_frame_sr_pngs')
        os.makedirs(pngs_dir, exist_ok=True)
        bb_det_file = os.path.join(pngs_parent_dir, str(output_height) + 'p_per_frame_sr_bbs')
        
        libvpx.get_png_of_raw(save_imgs_dir, output_width, output_height, pngs_dir)
        objdet.detect_objects(pngs_dir, bb_det_file)
        #### Mai, final task, end ####
        
        '''
        #### Mai, final task, nemo part, beg ####
        #create multiple processes for parallel quality measurements
        start_time = time.time()
        q0 = mp.Queue()
        q1 = mp.Queue()
        decoders = [mp.Process(target=libvpx.offline_cache_quality_mt, args=(q0, q1, self.vpxdec_path, self.dataset_dir, \
                                    self.lr_video_name, self.hr_video_name, self.model.name, self.output_width, self.output_height)) for i in range(self.num_decoders)]
        for decoder in decoders:
            decoder.start()

        #select a single anchor point and measure the resulting quality
        single_anchor_point_sets = []
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        for idx, frame in enumerate(frames):
            anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, frame.name)
            anchor_point_set.add_anchor_point(frame)
            anchor_point_set.save_cache_profile()
            q0.put((anchor_point_set.get_cache_profile_name(), num_skipped_frames, num_decoded_frames, postfix, idx))
            single_anchor_point_sets.append(anchor_point_set)
        for frame in frames:
            item = q1.get()
            idx = item[0]
            quality = item[1]
            single_anchor_point_sets[idx].set_measured_quality(quality)
            single_anchor_point_sets[idx].remove_cache_profile()

        #remove multiple processes
        for decoder in decoders:
            q0.put('end')
        for decoder in decoders:
            decoder.join()

        end_time = time.time()
        print('{} video chunk: (Step1-profile anchor point quality) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 2: order anchor points##########
        start_time = time.time()
        multiple_anchor_point_sets = []
        anchor_point_set = None
        FAST_anchor_point_set = single_anchor_point_sets[0]
        while len(single_anchor_point_sets) > 0:
            anchor_point_idx, estimated_quality = self._select_anchor_point(anchor_point_set, single_anchor_point_sets)
            selected_anchor_point = single_anchor_point_sets.pop(anchor_point_idx)
            if len(multiple_anchor_point_sets) == 0:
                anchor_point_set = libvpx.AnchorPointSet.load(selected_anchor_point, cache_profile_dir, '{}_{}'.format(algorithm_type, 1))
                anchor_point_set.set_estimated_quality(selected_anchor_point.measured_quality)
            else:
                anchor_point_set = libvpx.AnchorPointSet.load(multiple_anchor_point_sets[-1], cache_profile_dir, '{}_{}'.format(algorithm_type, multiple_anchor_point_sets[-1].get_num_anchor_points() + 1))
                anchor_point_set.add_anchor_point(selected_anchor_point.anchor_points[0])
                anchor_point_set.set_estimated_quality(estimated_quality)
            multiple_anchor_point_sets.append(anchor_point_set)

        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 3: select anchor points##########
        save_imgs_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix, 'nemo_sr')
        os.makedirs(save_imgs_dir, exist_ok=True)
        
        start_time = time.time()
        log_path0 = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        log_path1 = os.path.join(log_dir, 'quality_{}_8.txt'.format(algorithm_type))
        log_path2 = os.path.join(log_dir, 'quality_{}_16.txt'.format(algorithm_type))
        log_path3 = os.path.join(log_dir, 'quality_fast.txt')
        with open(log_path0, 'w') as f0, open(log_path1, 'w') as f1, open(log_path2, 'w') as f2, open(log_path3, 'w') as f3:
            for idx, anchor_point_set in enumerate(multiple_anchor_point_sets):
                #### Mai, final task, beg ####
                save_imgs = 1#None
                if len(multiple_anchor_point_sets) < 8:
                    if len(multiple_anchor_point_sets) < 8 and idx == len(multiple_anchor_point_sets) - 1:
                        save_imgs = 1
                elif idx == 8 - 1:
                    save_imgs = 1
                #### Mai, final task, end ####
                
                #log quality
                anchor_point_set.save_cache_profile()
                quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                                    num_skipped_frames, num_decoded_frames, postfix, \
                                                    save_imgs, save_imgs_dir) # Mai, final task, added
                
                anchor_point_set.remove_cache_profile()
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_point_set.get_num_anchor_points(), len(frames), \
                                         np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear), np.average(anchor_point_set.estimated_quality))
                f0.write(quality_log)
                if idx < 8:
                    f1.write(quality_log)
                if idx < 16:
                    f2.write(quality_log)
                if idx == 0:
                    f3.write(quality_log)
                print(np.average(quality_diff))
                #terminate
                if save_imgs == 1 and abs(np.average(quality_diff)) <= self.quality_margin:
                    break
                if abs(np.average(quality_diff)) <= self.quality_margin:
                    #### Mai, final task, beg ####
                    save_imgs = 1
                    anchor_point_set.save_cache_profile()
                    libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                                    num_skipped_frames, num_decoded_frames, postfix, save_imgs, save_imgs_dir)
                    anchor_point_set.remove_cache_profile()
                    
                    #### Mai, final task, end ####
                    #case 1: does not restrict #anchor points
                    anchor_point_set.set_cache_profile_name(algorithm_type)
                    anchor_point_set.save_cache_profile()

                    #case 2: limit #anchor points to 8
                    if anchor_point_set.get_num_anchor_points() > 8:
                        anchor_point_set_ = multiple_anchor_point_sets[7]
                    else:
                        anchor_point_set_ = anchor_point_set
                    anchor_point_set_.set_cache_profile_name('{}_8'.format(algorithm_type))
                    anchor_point_set_.save_cache_profile()

                    #case 3: limit #anchor points to 16
                    if anchor_point_set.get_num_anchor_points() > 16:
                        anchor_point_set_ = multiple_anchor_point_sets[15]
                    else:
                        anchor_point_set_ = anchor_point_set
                    anchor_point_set_.set_cache_profile_name('{}_16'.format(algorithm_type))
                    anchor_point_set_.save_cache_profile()

                    #case 4: FAST
                    anchor_point_set_ = FAST_anchor_point_set
                    anchor_point_set_.set_cache_profile_name('fast'.format(algorithm_type))
                    anchor_point_set_.save_cache_profile()

                    break

        end_time = time.time()
        print('{} video chunk: (Step3) {}sec'.format(chunk_idx, end_time - start_time))
        
        #### Mai, final task, beg ####
        #pngs_dir = os.path.join(pngs_parent_dir, str(output_height) + 'p_nemo_sr_pngs')
        #os.makedirs(pngs_dir, exist_ok=True)
        #bb_det_file = os.path.join(pngs_parent_dir, str(output_height) + 'p_nemo_sr_bbs')

        #libvpx.get_png_of_raw(save_imgs_dir, output_width, output_height, pngs_dir)
        #objdet.detect_objects(pngs_dir, bb_det_file)
        #### Mai, final task, end ####
        #### Mai, final task, nemo part, end ####
        '''
        
        #### Mai, final task, beg ####
        input_width, input_height = int(lr_video_profile['width']), int(lr_video_profile['height'])
        input_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        i_pngs_dir = os.path.join(pngs_parent_dir, str(input_height) + 'p_pngs')
        os.makedirs(i_pngs_dir, exist_ok=True)
        bb_det_file = os.path.join(pngs_parent_dir, str(input_height) + 'p_bbs')
        #bb_det_file = '/home/mai/nemo/data/nemo-data-traffic/traffic/experiment-outputs/1-filtered-small-240p-bbs/' + postfix + '/240p_bbs'
        libvpx.get_png_of_raw(input_dir, input_width, input_height, i_pngs_dir)
        objdet.detect_objects(i_pngs_dir, bb_det_file)
        shutil.copy(bb_det_file, os.path.join(input_dir, 'det_results')) # because they're used by super-resolution later
        #### Mai, final task, end ####
        
        start_time_bb_sr = time.time() # Mai, final task, added
        total_sr_time = libvpx.setup_sr_frame_nemo(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.model, postfix, is_dnn=False, scale=self.scale, sr_small_bbs_only=True) # Mai
        end_time_bb_sr = time.time() # Mai, final task, added
        
        #### Mai, final task, beg ####
        data_path = os.path.join(self.dataset_dir, 'detection_results', self.content, 'data.txt')
        with open(data_path, 'a') as data_file:
            data_file.write(postfix + '\n')
            data_file.write('1. DNN to all frames:\t\t\t\t\t\t{:.2f} sec\n'.format(end_time_dnn - start_time_dnn))
            data_file.write('2. DNN to bounding boxes in all frames (raw DNN time):\t\t{:.2f} sec\n'.format(total_sr_time))
            data_file.write('3. DNN to bounding boxes in all frames (includes all work):\t{:.2f} sec\n'.format(end_time_bb_sr - start_time_bb_sr))
        #### Mai, final task, end ####
        
        #create multiple processes for parallel quality measurements
        start_time = time.time()
        q0 = mp.Queue()
        q1 = mp.Queue()
        decoders = [mp.Process(target=libvpx.offline_cache_quality_mt, args=(q0, q1, self.vpxdec_path, self.dataset_dir, \
                                    self.lr_video_name, self.hr_video_name, self.model.name, self.output_width, self.output_height)) for i in range(self.num_decoders)]
        for decoder in decoders:
            decoder.start()
        
        #select a single anchor point and measure the resulting quality
        single_anchor_point_sets = []
        
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        for idx, frame in enumerate(frames):
            anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, frame.name)
            anchor_point_set.add_anchor_point(frame)
            anchor_point_set.save_cache_profile()
            q0.put((anchor_point_set.get_cache_profile_name(), num_skipped_frames, num_decoded_frames, postfix, idx))
            single_anchor_point_sets.append(anchor_point_set)
            
        for frame in frames:
            item = q1.get()
            idx = item[0]
            quality = item[1]
            single_anchor_point_sets[idx].set_measured_quality(quality)
            single_anchor_point_sets[idx].remove_cache_profile()

        #remove multiple processes
        for decoder in decoders:
            q0.put('end')
        for decoder in decoders:
            decoder.join()

        end_time = time.time()
        print('{} video chunk: (Step1-profile anchor point quality) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 2: order anchor points##########
        start_time = time.time()
        multiple_anchor_point_sets = []
        anchor_point_set = None
        FAST_anchor_point_set = single_anchor_point_sets[0]
        while len(single_anchor_point_sets) > 0:
            anchor_point_idx, estimated_quality = self._select_anchor_point(anchor_point_set, single_anchor_point_sets)
            selected_anchor_point = single_anchor_point_sets.pop(anchor_point_idx)
            if len(multiple_anchor_point_sets) == 0:
                anchor_point_set = libvpx.AnchorPointSet.load(selected_anchor_point, cache_profile_dir, '{}_{}'.format(algorithm_type, 1))
                anchor_point_set.set_estimated_quality(selected_anchor_point.measured_quality)
            else:
                anchor_point_set = libvpx.AnchorPointSet.load(multiple_anchor_point_sets[-1], cache_profile_dir, '{}_{}'.format(algorithm_type, multiple_anchor_point_sets[-1].get_num_anchor_points() + 1))
                anchor_point_set.add_anchor_point(selected_anchor_point.anchor_points[0])
                anchor_point_set.set_estimated_quality(estimated_quality)
            multiple_anchor_point_sets.append(anchor_point_set)

        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 3: select anchor points##########
        
        #### Mai, final task, beg ####
        save_imgs_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix, 'per_bb_sr')
        os.makedirs(save_imgs_dir, exist_ok=True)    
        for f in listdir(i_pngs_dir):
            if '_' in f:
                os.remove(os.path.join(i_pngs_dir, f))
        #### Mai, final task, end  ####
        
        start_time = time.time()
        log_path0 = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        log_path1 = os.path.join(log_dir, 'quality_{}_8.txt'.format(algorithm_type))
        log_path2 = os.path.join(log_dir, 'quality_{}_16.txt'.format(algorithm_type))
        log_path3 = os.path.join(log_dir, 'quality_fast.txt')
        with open(log_path0, 'w') as f0, open(log_path1, 'w') as f1, open(log_path2, 'w') as f2, open(log_path3, 'w') as f3:
            for idx, anchor_point_set in enumerate(multiple_anchor_point_sets):
                #### Mai, final task, beg ####
                save_imgs = None
                if len(multiple_anchor_point_sets) < 8:
                    if len(multiple_anchor_point_sets) < 8 and idx == len(multiple_anchor_point_sets) - 1:
                        save_imgs = 1
                elif idx == 8 - 1:
                    save_imgs = 1
                #### Mai, final task, end ####
                
                #log quality
                anchor_point_set.save_cache_profile()
                quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                                    self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                                    num_skipped_frames, num_decoded_frames, postfix, \
                                                    save_imgs, save_imgs_dir) # Mai, final task, added
                anchor_point_set.remove_cache_profile()
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_point_set.get_num_anchor_points(), len(frames), \
                                         np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear), np.average(anchor_point_set.estimated_quality))
                f0.write(quality_log)
                if idx < 8:
                    f1.write(quality_log)
                if idx < 16:
                    f2.write(quality_log)
                if idx == 0:
                    f3.write(quality_log)
                print('average quality diff:', np.average(quality_diff))
                #terminate
                if save_imgs == 1:
                    break
        end_time = time.time()
        print('{} video chunk: (Step3) {}sec'.format(chunk_idx, end_time - start_time))
        
        #### Mai, final task, beg ####
        pngs_dir = os.path.join(pngs_parent_dir, str(output_height) + 'p_per_bb_sr_pngs')
        os.makedirs(pngs_dir, exist_ok=True)
        bb_det_file = os.path.join(pngs_parent_dir, str(output_height) + 'p_per_bb_sr_bbs')

        libvpx.get_png_of_raw(save_imgs_dir, output_width, output_height, pngs_dir)
        objdet.detect_objects(pngs_dir, bb_det_file)
        #### Mai, final task, end ####
        
        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

    #select an anchor point set whose anchor points are uniformly located
    def _select_anchor_point_set_uniform(self, chunk_idx=None):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)
        algorithm_type = 'uniform_{}'.format(self.quality_margin)

        ###########step 1: measure bilinear, dnn quality##########
        #calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = get_video_profile(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        #save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx.setup_sr_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.model, postfix)

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        quality_dnn = libvpx.offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                 self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        end_time = time.time()
        print('{} video chunk: (Step1-profile bilinear, dnn quality) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 2: select anchor points##########
        start_time = time.time()
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        log_path = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        with open(log_path, 'w') as f:
            for i in range(len(frames)):
                #select anchor point uniformly
                num_anchor_points = i + 1
                anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, '{}_{}'.format(algorithm_type, num_anchor_points))
                for j in range(num_anchor_points):
                    idx = j * math.floor(len(frames) / num_anchor_points)
                    anchor_point_set.add_anchor_point(frames[idx])

                #measure the quality
                anchor_point_set.save_cache_profile()
                quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                        self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                        num_skipped_frames, num_decoded_frames, postfix)
                anchor_point_set.remove_cache_profile()
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_point_set.get_num_anchor_points(), len(frames), \
                                         np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))
                f.write(quality_log)

                #terminate
                if np.average(quality_diff) <= self.quality_margin:
                    anchor_point_set.set_cache_profile_name(algorithm_type)
                    anchor_point_set.save_cache_profile()
                    libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                            self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                            num_skipped_frames, num_decoded_frames, postfix)
                    break

        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))

        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

    #select an anchor point set whose anchor points are randomly located
    def _select_anchor_point_set_random(self, chunk_idx=None):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)
        algorithm_type = 'random_{}'.format(self.quality_margin)

        ###########step 1: measure bilinear, dnn quality##########
        #calculate num_skipped_frames and num_decoded frames
        start_time = time.time()
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = get_video_profile(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        assert(num_total_frames == math.floor(num_total_frames))
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        #save low-resolution, super-resoluted, high-resolution frames to local storage
        libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
        libvpx.setup_sr_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.model, postfix)

        #measure bilinear, per-frame super-resolution quality
        quality_bilinear = libvpx.bilinear_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.output_width, self.output_height,
                                                    num_skipped_frames, num_decoded_frames, postfix)
        quality_dnn = libvpx.offline_dnn_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, \
                                                 self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)

        end_time = time.time()
        print('{} video chunk: (Step1-profile bilinear, dnn quality) {}sec'.format(chunk_idx, end_time - start_time))

        ###########step 2: select anchor points##########
        start_time = time.time()
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
        log_path = os.path.join(log_dir, 'quality_{}.txt'.format(algorithm_type))
        with open(log_path, 'w') as f:
            for i in range(len(frames)):
                #select anchor point uniformly
                num_anchor_points = i + 1
                anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, '{}_{}'.format(algorithm_type, num_anchor_points))
                random_frames = random.sample(frames, num_anchor_points)
                for frame in random_frames:
                    anchor_point_set.add_anchor_point(frame)

                #measure the quality
                anchor_point_set.save_cache_profile()
                quality_cache = libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                        self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                        num_skipped_frames, num_decoded_frames, postfix)
                anchor_point_set.remove_cache_profile()
                quality_diff = np.asarray(quality_dnn) - np.asarray(quality_cache)
                quality_log = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(anchor_point_set.get_num_anchor_points(), len(frames), \
                                         np.average(quality_cache), np.average(quality_dnn), np.average(quality_bilinear))
                f.write(quality_log)

                #terminate
                if np.average(quality_diff) <= self.quality_margin:
                    anchor_point_set.set_cache_profile_name(algorithm_type)
                    anchor_point_set.save_cache_profile()
                    libvpx.offline_cache_quality(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, \
                                            self.model.name, anchor_point_set.get_cache_profile_name(), self.output_width, self.output_height, \
                                            num_skipped_frames, num_decoded_frames, postfix)
                    break

        end_time = time.time()
        print('{} video chunk: (Step2) {}sec'.format(chunk_idx, end_time - start_time))

        #remove images
        lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
        hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
        sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
        shutil.rmtree(lr_image_dir, ignore_errors=True)
        shutil.rmtree(hr_image_dir, ignore_errors=True)
        shutil.rmtree(sr_image_dir, ignore_errors=True)

    def select_anchor_point_set(self, algorithm_type, chunk_idx=None, max_nemo_num_anchor_points=None):
        if chunk_idx is not None:
            if algorithm_type == 'nemo':
                self._select_anchor_point_set_nemo(chunk_idx)
            elif algorithm_type == 'uniform':
                self._select_anchor_point_set_uniform(chunk_idx)
            elif algorithm_type == 'random':
                self._select_anchor_point_set_random(chunk_idx)
        else:
            lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
            lr_video_profile = get_video_profile(lr_video_path)
            num_chunks = int(math.ceil(lr_video_profile['duration'] / (self.gop / lr_video_profile['frame_rate'])))
            
            #### Mai, final task, beg ####
            pngs_parent_dir = os.path.join(self.dataset_dir, 'detection_results', self.content)
            os.makedirs(pngs_parent_dir, exist_ok=True)
            data_path = os.path.join(pngs_parent_dir, 'data.txt')
            with open(data_path, 'w') as data_file:
                data_file.write('Total number of chunks:\t{}\n\n'.format(num_chunks))
            #### Mai, final task, end ####
            
            for i in range(num_chunks):
                if algorithm_type == 'nemo':
                    self._select_anchor_point_set_nemo(i)
                elif algorithm_type == 'uniform':
                    self._select_anchor_point_set_uniform(i)
                elif algorithm_type == 'random':
                    self._select_anchor_point_set_random(i)

    def _aggregate_per_chunk_results(self, algorithm_type):
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = get_video_profile(lr_video_path)
        num_chunks = int(math.ceil(lr_video_profile['duration'] / (self.gop / lr_video_profile['frame_rate'])))
        start_idx = 0
        end_idx = num_chunks - 1

        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name)
        log_name = os.path.join('quality_{}.txt'.format(algorithm_type))
        cache_profile_name = os.path.join('{}.profile'.format(algorithm_type))

        #log
        log_path = os.path.join(log_dir, log_name)
        with open(log_path, 'w') as f0:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                chunk_log_dir = os.path.join(log_dir, 'chunk{:04d}'.format(chunk_idx))
                chunk_log_path= os.path.join(chunk_log_dir, log_name)
                with open(chunk_log_path, 'r') as f1:
                    q_lines = f1.readlines()
                    f0.write('{}\t{}\n'.format(chunk_idx, q_lines[-1].strip()))

        #cache profile
        cache_profile_path = os.path.join(cache_profile_dir, cache_profile_name)
        cache_data = b''
        with open(cache_profile_path, 'wb') as f0:
            for chunk_idx in range(start_idx, end_idx + 1):
                chunk_cache_profile_path = os.path.join(cache_profile_dir, 'chunk{:04d}'.format(chunk_idx), cache_profile_name)
                with open(chunk_cache_profile_path, 'rb') as f1:
                    f0.write(f1.read())

        #log (bilinear, sr)
        log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, 'quality.txt')
        with open(log_path, 'w') as f0:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                quality = []
                chunk_log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, 'chunk{:04d}'.format(chunk_idx), 'quality.txt')
                with open(chunk_log_path, 'r') as f1:
                    lines = f1.readlines()
                    for line in lines:
                        line = line.strip()
                        quality.append(float(line.split('\t')[1]))
                    f0.write('{}\t{:.4f}\n'.format(chunk_idx, np.average(quality)))

        log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, 'quality.txt')
        with open(log_path, 'w') as f0:
            #iterate over chunks
            for chunk_idx in range(start_idx, end_idx + 1):
                quality = []
                chunk_log_path = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, 'chunk{:04d}'.format(chunk_idx), 'quality.txt')
                with open(chunk_log_path, 'r') as f1:
                    lines = f1.readlines()
                    for line in lines:
                        line = line.strip()
                        quality.append(float(line.split('\t')[1]))
                    f0.write('{}\t{:.4f}\n'.format(chunk_idx, np.average(quality)))

    def aggregate_per_chunk_results(self, algorithm_type):
        if algorithm_type == 'nemo':
            self._aggregate_per_chunk_results('{}_{}'.format(algorithm_type, self.quality_margin))
            self._aggregate_per_chunk_results('{}_{}_8'.format(algorithm_type, self.quality_margin))
            self._aggregate_per_chunk_results('{}_{}_16'.format(algorithm_type, self.quality_margin))
            self._aggregate_per_chunk_results('fast')
        else:
            self._aggregate_per_chunk_results('{}_{}'.format(algorithm_type, self.quality_margin))

