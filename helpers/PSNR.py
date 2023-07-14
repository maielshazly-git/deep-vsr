import os
import numpy as np

PSNR = []
for i in range(0, 9):
    with open('/home/mai/nemo/data/nemo-data-traffic/traffic/detection_results/traffic/chunk000' + str(i) + '/1080p_per_bb_sr_pngs/' + 'quality.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            PSNR.append(float(line.split('\t')[1]))
    print('Chunk ' + str(i), ':\t', round(np.average(PSNR), 2))
