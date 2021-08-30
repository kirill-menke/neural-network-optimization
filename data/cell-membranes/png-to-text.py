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


def calculate_mean(images_dir):
    images = []
    for name in os.listdir(images_dir):
        path = f"{images_dir}/{name}"
        image = np.array(plt.imread(path))

        images.append(image)

    print("mean: ", np.mean(np.asarray(images)))
    print("std: ", np.std(np.asarray(images)))


img2text('./train-images-small', 'train-', normalize=False)
img2text('./train-labels-small', 'label-', normalize=False)



