import pandas as pd
import numpy as np
import os
import scipy.io as sio


def main():

    save_feature_vectors2txt()


def save_feature_vectors2txt():
    embedding_vectors = sio.loadmat("../data/link_prediction/ggi_0.8_unweighted.mat")
    embedding_vectors = embedding_vectors['embedding']
    # 12331 200
    cnt = 0
    line_number = True
    with open("../pre_train/link_prediction/ggi_0.8_unweighted_pre_train.emb", 'w') as out:
        out.write("%d\t%d\n" % (12331, 200))
        for features in embedding_vectors:
            if line_number:
                out.write("%d\t" % cnt)
            for dim in features:
                out.write("%.20f\t" % dim)
            out.write("\n")
            cnt += 1
    print(cnt)


if __name__ == "__main__":
    main()
