# neural-sr-for-bbs

This research project is based on the [nemo](https://github.com/kaist-ina/nemo) project.
The goal of this project is to compare the outputs of different experiments that apply neural super-resolution to videos on mobile devices to acquire bounding boxes that can be acceptable as ground truth for possible further object detection on such devices.

## Prerequisites

1. Follow all installation steps and commands in README.md of [nemo](https://github.com/kaist-ina/nemo) until step (4. Generate a cache profile) inclusive.
2. The following libraries are required for the object detector to work:

```
conda install python=3.9.0 pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

conda install python=3.9.0 networkx imageio tensorflow-gpu==1.15

conda install python=3.9.0 matplotlib -c conda-forge
```

3. The object detector also needs the trained weights of its backbone. The weights can be downloaded from [Kaggle](https://www.kaggle.com/datasets/n1t1nk/fasterrcnn-resnet50-fpn-coco?resource=download+%E2%80%8B). Place the downloaded file inside object_detector/Backend/backend, then update the location in the \_\_init\_\_() function in [object_detector/Backend/backend/object_detector.py](object_detector/Backend/backend/object_detector.py) accordingly.
 
## Experiments

Please note that this project changes some code parts of the [nemo](https://github.com/kaist-ina/nemo) project and its sr-integrated codec. Aside from the README.md of nemo, all explanation below refers to the code present in this repository.

To run any of the experiments, you can simply run the command provided by [nemo](https://github.com/kaist-ina/nemo)

```
$NEMO_CODE_ROOT/nemo/cache_profile/script/select_anchor_points.sh -g 0 -c product_review -q high -i 240 -o 1080 -a nemo
```
after changing the relevant parts in the code, depending on which experiment you want to test.

If you want to work on a video of your own, upload it to YouTube then add its URL along with an indicating keyword to [nemo/tool/video.py](nemo/tool/video.py).

### Relevant directories

1. You should find the frames resulting from upscaling using the SR DNN, i.e., per-frame SR, in the directory:

```
[dataset_dir]/image/[lr_video_name]/[postfix]/per_frame_sr

E.g., $NEMO_DATA_ROOT/traffic/image/240p_512kbps_s0_d300.webm/chunk0000/per_frame_sr
```

These frames are saved at some point within the sr-integrated codec present in [third_party/libvpx](third_party/libvpx), which saves the frames in RAW format.

2. These frames are converted to PNGs and saved in the directory:

```
[dataset_dir]/detection_results/[content]/[postfix]/[str(output_height) + 'p_per_frame_sr_pngs']	

E.g., $NEMO_DATA_ROOT/traffic/detection_results/traffic/chunk0000/1080p_per_frame_sr_pngs
```
	
And the results of the object detector for these frames can be found in the file:

```
[dataset_dir]/detection_results/[content]/[postfix]/[str(output_height) + 'p_per_frame_sr_bbs']

E.g., $NEMO_DATA_ROOT/traffic/detection_results/traffic/chunk0000/1080p_per_frame_sr_bbs
```

3. Similarly, the PNG frames of the input video without upscaling, 240p in the experiments, can be found in the directory:

```
[dataset_dir]/detection_results/[content]/[postfix]/[str(input_height) + 'p_pngs']

E.g., $NEMO_DATA_ROOT/traffic/detection_results/traffic/chunk0000/240p_pngs
```
	
And the results of the object detector of these frames can be found in the file:

```
[dataset_dir]/detection_results/[content]/[postfix]/[str(input_height) + 'p_bbs']

E.g., $NEMO_DATA_ROOT/traffic/detection_results/traffic/chunk0000/240p_bbs
```
	
4. The DNN time taken by the per-frame SR part as well as that taken by the AP Per-BB SR/AP Per-Small-BB SR experiment is measured and saved in the file:

```
[dataset_dir]/detection_results/[content]/data.txt

E.g., $NEMO_DATA_ROOT/traffic/detection_results/traffic/data.txt
```

There, you will find the raw DNN time as well as the total DNN processing time taken by the AP Per-BB SR/AP Per-Small-BB SR experiment, depending on which experiment you're running at the moment. To understand what the two time terms refer to, please look at [Presentation.pdf](Presentation.pdf). The time is measured in seconds. The total number of chunks in the video is also written at the beginning of data.txt.

5. Similar to (1), the RAW frames resulting from the AP Per-BB SR/AP Per-Small-BB SR experiment are saved in the directory:

```
[dataset_dir]/image/[lr_video_name]/[postfix]/per_bb_sr

E.g., $NEMO_DATA_ROOT/traffic/image/240p_512kbps_s0_d300.webm/chunk0000/per_bb_sr
```

6. The PNGs of these frames are saved in the directory:

```
[dataset_dir]/detection_results/[content]/[postfix]/[str(output_height) + 'p_per_bb_sr_pngs']	

E.g., $NEMO_DATA_ROOT/traffic/detection_results/traffic/chunk0000/1080p_per_bb_sr_pngs
```
	
And their object detection results are saved in the file:

```
[dataset_dir]/detection_results/[content]/[postfix]/[str(output_height) + 'p_per_bb_sr_bbs']

E.g., $NEMO_DATA_ROOT/traffic/detection_results/traffic/chunk0000/1080p_per_bb_sr_bbs
```
	
7. If you decide to run the AP SR (NEMO) experiment, simply uncomment the code snippet starting from the comment (Mai,final task, nemo part, beg) until (Mai, final task, nemo part, end) and comment the code remaining in the function until before the comment (remove images).

8. The PNGs resulting from the AP SR (NEMO) experiment are saved in the directory:

```
[dataset_dir]/detection_results/[content]/[postfix]/[str(output_height) + 'p_nemo_sr_pngs']	

E.g., $NEMO_DATA_ROOT/traffic/detection_results/traffic/chunk0000/1080p_nemo_sr_pngs
```

And their object detection results are saved in the file:

```
[dataset_dir]/detection_results/[content]/[postfix]/[str(output_height) + 'p_nemo_sr_bbs']

E.g., $NEMO_DATA_ROOT/traffic/detection_results/traffic/chunk0000/1080p_nemo_sr_bbs
```
### Relevant explanation about the parameters of the experiments

* In the AP Per-BB SR and AP Per-Small-BB SR experiments, bounding boxes are filtered by their confidence scores, so only objects with at least 50% confidence are considered.

* In case you want to try out different thresholds to filter at, go to object_detector/Backend/detect_fn.py and update the score used for filtering; it's 0.5 now. You'd find a guiding comment at the intended comment saying (added to filter out less confident BBs). Doing this will cause the object detector to return filtered bounding boxes in the first place.

* By default, the code should run the AP Per-BB SR experiment. However, if you want to run the AP Per-Small-BB SR experiment, uncomment the relevant part in [nemo/tool/libvpx.py](nemo/tool/libvpx.py). The relevant part is right below the comment (optional task, sr only small bbs). The AP Per-Small-BB SR experiment by default considers objects with bounding boxes smaller than 20x20 pixels.

* In case you want to test different area thresholds based on the percentage of their bounding boxes among all frames, first try out different area values in [helpers/small_bbs_percent.py](helpers/small_bbs_percent.py) and see which area/percentage you'd like to test.

* Use [helpers/draw_bbs.py](helpers/draw_bbs.py) in case you want to visualize bounding boxes in their corresponding frames. You need to specify the intended chunk number to the __postfix__ variable. Visualization is done for all frames in the specified chunk, along with object labels for bounding boxes and frame numbers.

* By default, visualization is done for all filtered bounding boxes; in case you're testing the AP Per-Small-BB SR experiment and want to only visualize small bounding boxes, uncomment the lines after the comment (To visualize only small bbs). Remember to update the small area threshold in case you decided to work with a different value. It's by default 20x20.

* You can visualize bounding boxes for any of the experiments. To do so, change the directory/file name provided for the __bbs_file__ and the __pngs_dir__ variables accordingly. You can find all options in comments next to each. By default, visualizations are done on the 240p frames.

* Although PSNR values are irrelevant for the experiments since enhanced quality is not one of the objectives, you can still find the PSNR values for any of the chunks for the current experiment in the log directory. This part is already handled by the [nemo](https://github.com/kaist-ina/nemo) code unchanged.

* [helpers/PSNR.py](helpers/PSNR.py) simply computes the average PSNR for a chunk, provided the directory of the quality.txt file containing the PSNR values of the frames in that chunk.

* [helpers/f1_score.py](helpers/f1_score.py) contains the code responsible for computing the accuracy/f1-scores for each experiment.

* Since it computes the f1-score for many experiments at once, you need to arrange files in the same hierarchy and naming as that found in the [drive folder](https://drive.google.com/drive/folders/1pV6MazsXyauXYQPzNF0vMfOxu8nEDzJ3?usp=sharing) mentioned below. The expected structure of the results of running [helpers/f1_score.py](helpers/f1_score.py) is the same as the structure of the directory accuracy_results in the [drive folder](https://drive.google.com/drive/folders/1pV6MazsXyauXYQPzNF0vMfOxu8nEDzJ3?usp=sharing).

### Google Drive folder with results

You can already find the latency, quality, object detection results, and the number of anchor points for all chunks as well many frame sample results in [this drive folder](https://drive.google.com/drive/folders/1pV6MazsXyauXYQPzNF0vMfOxu8nEDzJ3?usp=sharing) for many experiments. Following is a brief explanation on what its directory names refer to:

__0-1080p---bilinear:__

PSNR/Quality values in case of upscaling all frames using bilinear interpolation.
	
__1-240p:__

Results after working on the input low-resolution frames.
	
__2-1080p---per-frame-sr:__

Results after applying per-frame SR.
	
__3-mis---1080p---nemo-sr---from-100th-ap---max-8-aps:__

A side experiment. Originally, the sr-integrated codec provided by [nemo](https://github.com/kaist-ina/nemo) selects anchor points starting from the 100th frame in a chunk, in case of the offline mode, which is the mode used in this research project.

The quality of the resulting frames is quite worse than those of the below experiment, but it was interesting to see the influence of the order of the anchor point frames on the overall results, so this directory contains the results.

This is changed in the sr-integrated codec here. To test what the original results would look like, you can see the frame samples in this directory. However, if you're working on your own video, you can test this through the following steps:

a. Uncomment the code part below the comment ((deprecated) NEMO: fullfill buffer at the beginning of video streaming) in [third_party/libvpx/vp9/decoder/vp9_decodeframe.c](third_party/libvpx/vp9/decoder/vp9_decodeframe.c).

b. Change your directory to that of the sr-integrated codec:

```
cd third_party/libvpx
```

c. Run the following commands to update the binary file:

```
sh ./configure

make -f Makefile
```
d. The new binary file is called vpxdec_nemo_ver2, copy it and paste it in the bin directory to replace the old one.
	
__4-1080p---nemo-sr---any-ap---no-max-aps:__

This refers to the AP SR (NEMO) experiment. Please refer to [Presentation.pdf](Presentation.pdf) to understand the main experiments with their names. This is also considered a fix to the above experiment.
	
__5-1080p---per-bb-sr---any-ap---max-8-aps:__

This refers to the AP Per-BB SR experiment with a maximum of 8 anchor points. To know why providing a limit for the anchor points was necessary, please refer to [Presentation.pdf](Presentation.pdf).
	
__6-1080p---per-bb-sr---any-ap---max-16-aps:__

This refers to the AP Per-BB SR experiment with a maximum of 16 anchor points. This limit can be changed in [nemo/cache_profile/anchor_point_selector.py](nemo/cache_profile/anchor_point_selector.py) in the ___select_anchor_point_set_nemo__ function.
	
__7-1080p---per-small-bb-sr---any-ap---max-8-aps:__

This refers to the AP Per-Small-BB SR experiment with a maximum of 8 anchor points.

