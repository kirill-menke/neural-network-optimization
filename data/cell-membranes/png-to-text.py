#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os

normalize = False
images_dir = './train-images'
output_prefix = 'train-'

for name in os.listdir(images_dir):
    path = f"{images_dir}/{name}"
    print(path)
    image = np.array(plt.imread(path))

    if normalize:
        image -= np.average(image)

    with open(f"./{output_prefix}{name}.txt", "w") as f:
        for row in image:
            for pixel in row:
                print(f"\t{pixel:.3f}", file=f, end='')
            print('', file=f)

