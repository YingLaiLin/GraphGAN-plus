# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.io as sio
from tqdm import tqdm


def emb2mat(filename, sep="\t"):
    feature_vectors = []
    try:
        print("reading {} ......".format(filename))
        with open(filename) as f:
            nodes, dim = f.readline().split()  # ignore
            for _ in tqdm(range(int(nodes))):
                line = f.readline()

                vectors = np.array(line.split()).astype(float)
                feature_vectors.append(vectors)

    except IOError:
        print("{} is not exist!Please check the path of file".format(filename))
    name = filename[filename.index("ggi"):filename.index(".emb")]

    sio.savemat(name + '.mat', {
        'features': np.array(feature_vectors), 'name': filename
    })


if __name__ == "__main__":
    prefix = '/Users/leon/research/code/GraphGAN/pre_train/link_prediction/'
    filenames = ["ggi_0.8_unweighted_dis.emb",
                 "ggi_0.8_unweighted_gen.emb"]
    for filename in filenames:
        emb2mat(prefix + filename)
