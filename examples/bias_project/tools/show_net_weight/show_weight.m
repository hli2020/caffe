clc; close all; clear;

dataset_path = '/home/hyli/dataset/cifar';
test_batch = load('/home/hyli/dataset/cifar_matlab/test_batch.mat');
images = test_batch.data;
labels = test_batch.labels;

% returns mean_data in W x H x C with BGR channels
mean_data = caffe.io.read_mean([dataset_path '/cifar10_mean.binaryproto']);

which_im = 13;
im = reshape(images(which_im, :), [32, 32, 3]);
im = permute(im, [2, 1, 3]);
%imshow(im);
% preprocess
im_ = im(:, :, [3,2,1]);
im_ = permute(im_, [2,1,3]);
im_ = single(im_);
im_ = im_ - mean_data;

caffe.reset_all();
gpu_id = 0;
phase = 'test';

net_weights = '../../v9/cifar10/cifar10_v9.0_iter_58000_id=6.caffemodel';
net_def = '../../v9/cifar10/net_def/train_val_v9.0_neat_deploy.prototxt';
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

% Initialize a network
net = caffe.Net(net_def, net_weights, phase);
% do forward
res = net.forward({im_});
prob = res{1};

% conv1_1_w = net.params('conv1_1', 1).get_data();
% conv1_1_b = net.params('conv1_1', 2).get_data();

conv1_1_feat = net.blobs('conv1_1_bn').get_data();
pool2_feat = net.blobs('pool2').get_data();


