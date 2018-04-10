# -*- coding: utf-8 -*-

batch_size_dis = 64  # batch size for discriminator
batch_size_gen = 64  # batch size for generator
lambda_dis = 1e-5  # l2 loss regulation factor for discriminator
lambda_gen = 1e-5  # l2 loss regulation factor for generator
n_sample_dis = 20  # sample num for generator
n_sample_gen = 20  # sample num for discriminator
update_ratio = 1  # updating ratio when choose the trees
save_steps = 10

lr_dis = 1e-4  # learning rate for discriminator
lr_gen = 1e-3  # learning rate for discriminator

max_epochs = 20  # outer loop number
max_epochs_gen = 30  # loop number for generator
max_epochs_dis = 30  # loop number for discriminator

gen_for_d_iters = 10  # iteration numbers for generate new data for discriminator
max_degree = 0  # the max node degree of the network
model_log_dir = "/log/"

use_mul = False  # control if use the multiprocessing when constructing trees
load_model = False  # if load the model for continual training
gen_update_iter = 200
window_size = 3

app = "link_prediction"
source_data = "/ggi_0.8_unweighted_"
# source_data = "/test_"
train_filename = "../../data/" + app + source_data + "train.txt"
test_filename = "../../data/" + app + source_data + "test.txt"
test_neg_filename = "../../data/" + app + source_data + "test_neg.txt"
prob_filename = "../../data/" + app + source_data + "probs.csv"
n_node = 12331
n_embed = 200
# fraction of softmax
fraction_of_softmax = 0.5

# pre_train embbeding result
pretrain_emd_filename_d = "../../pre_train/" + app + source_data + "pre_train.emb"
pretrain_emd_filename_g = "../../pre_train/" + app + source_data + "pre_train.emb"
modes = ["dis", "gen"]
emb_filenames = [
    "../../pre_train/" + app + source_data + modes[0] + ".emb",
    "../../pre_train/" + app + source_data + modes[1] + ".emb"]
result_filename = "../../results/" + app + source_data + "res.txt"
