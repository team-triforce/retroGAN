"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import shutil
import time

import json
from numpy import rad2deg
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.util import mkdir
from util.visualizer import Visualizer
from util import evaluation_metrics, random_search
import os
from os.path import join, isdir, exists
from PIL import Image
import uuid

# CHECKPOINT_BASE_PATH = r'./checkpoints/triforce/web/images'


def main(opt):
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    if len(opt.random_search) > 0:
        return compute_metrics()


def compute_metrics():
    # get all generated images
    filtered_images = filter(lambda filename: "fake" in filename, os.listdir(CHECKPOINT_IMAGE_PATH))
    nes_color_ratio = 0
    snes_color_ratio = 0
    # get images from last epoch
    sorted_images = sorted(filtered_images)[-2:]
    for filename in sorted_images:
        img = Image.open(join(CHECKPOINT_IMAGE_PATH, filename))
        # for now assume fake_A is nes and fake_B is snes
        if "A" in filename:
            nes_color_ratio = evaluation_metrics.compute_nes_color_ratio(img)
        else:
            snes_color_ratio = evaluation_metrics.compute_snes_color_ratio(img)
    print(f'Ratio of correct NES colors: {nes_color_ratio}')
    print(f'Ratio of correct SNES colors: {snes_color_ratio}')
    return nes_color_ratio, snes_color_ratio


def clear_checkpoint_images():
    # clear checkpoints images folder
    if isdir(CHECKPOINT_IMAGE_PATH):
        for filename in os.listdir(CHECKPOINT_IMAGE_PATH):
            os.remove(join(CHECKPOINT_IMAGE_PATH, filename))


def copyfile(file, src_dir, dst_dir):
    src_path = os.path.join(src_dir, file)
    if exists(src_path):
        mkdir(dst_dir)
        shutil.copy(src_path, dst_dir)


def run_random_search(opt):
    best_nes_opt = None
    best_nes_metric = None
    best_nes_ix = None
    best_snes_opt = None
    best_snes_metric = None
    best_snes_ix = None

    # create random sets of opts, and train over each one, retaining the best
    random_opts = random_search.get_randomized_opts(opt)
    random_opt_dict = {}
    for ix, random_opt in enumerate(random_opts):
        print('Running RANDOM SEARCH with:')
        print(random_opt)
        clear_checkpoint_images()
        try:
        # if True:
            candidate_nes_metric, candidate_snes_metric = main(random_opt)

            dest_dir = str(uuid.uuid4()).replace('-', '')
            out_dir = os.path.join(CHECKPOINT_BASE_PATH, dest_dir)
            src_dir = f'{CHECKPOINT_BASE_PATH}'
            # back up models
            copyfile('latest_net_D_A.pth', src_dir=src_dir, dst_dir=out_dir)
            copyfile('latest_net_D_B.pth', src_dir=src_dir, dst_dir=out_dir)
            copyfile('latest_net_G_A.pth', src_dir=src_dir, dst_dir=out_dir)
            copyfile('latest_net_G_B.pth', src_dir=src_dir, dst_dir=out_dir)
            copyfile('log_loss.txt', src_dir=src_dir, dst_dir=out_dir)
            copyfile('train_opt.txt', src_dir=src_dir, dst_dir=out_dir)

            random_opt_dict[ix] = {
                'model_directory': dest_dir,
                'nes_metric': candidate_nes_metric,
                'snes_metric': candidate_snes_metric,
                'opts': random_opt.__dict__
            }

            # save off checkpoint
            if best_nes_metric is None or candidate_nes_metric > best_nes_metric:
                best_nes_metric = candidate_nes_metric
                best_nes_opt = opt
                best_nes_ix = ix

            if best_snes_metric is None or candidate_snes_metric > best_snes_metric:
                best_snes_metric = candidate_snes_metric
                best_snes_opt = opt
                best_snes_ix = ix
        except:
            print("error during last run")
            pass

    # write output
    random_opt_dict['best_nes_id'] = best_nes_ix
    random_opt_dict['best_snes_id'] = best_snes_ix

    out_filename = os.path.join(CHECKPOINT_BASE_PATH, 'results.json')
    with open(out_filename, "w") as outfile:
        json.dump(random_opt_dict, outfile, indent=4)

    print(f'Best NES metric score: {best_nes_metric}')
    print('Best opt:')
    if best_nes_opt is not None:
        best_nes_opt = best_nes_opt.__dict__
        for o in best_nes_opt:
            print(f'{o}: {best_nes_opt[o]}')

    print(f'Best SNES metric score: {best_snes_metric}')
    print('Best opt:')
    if best_snes_opt is not None:
        best_snes_opt = best_snes_opt.__dict__
        for o in best_snes_opt:
            print(f'{o}: {best_snes_opt[o]}')


if __name__ == '__main__':
    options = TrainOptions().parse()  # get training options

    CHECKPOINT_BASE_PATH = f'{options.checkpoints_dir}/{options.name}'
    CHECKPOINT_WEB_PATH = f'{CHECKPOINT_BASE_PATH}/web'
    CHECKPOINT_IMAGE_PATH = f'{CHECKPOINT_WEB_PATH}/images'

    if exists(options.random_search):
        run_random_search(options)
    else:
        main(options)
