#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

def img2text(images_dir, output_prefix, normalize):
    for name in os.listdir(images_dir):
        path = f"{images_dir}/{name}"
        out = f"./{output_prefix}{name}.txt"
        print(f"{path} -> {out}")

        image = np.array(plt.imread(path))

        if normalize:
            image -= np.average(image)

        with open(out, "w") as f:
            for row in image:
                for pixel in row:
                    print(f"\t{pixel:.5f}", file=f, end='')
                print('', file=f)

img2text('./train-images', 'train-', normalize=False)
img2text('./train-labels', 'label-', normalize=False)

