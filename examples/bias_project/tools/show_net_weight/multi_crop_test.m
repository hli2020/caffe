clc; close all;
%clear;
caffe.reset_all();
gpu_id = 0;
phase = 'test';
% net_weights = '../../v8/result_8.3/cifar100_aug_iter_186000.caffemodel';
% net_def = '../../v8/train_val_v8.3_cifar100_aug_deploy.prototxt';
net_weights = '../../v9/cifar10/cifar10_v9.0_iter_58000_id=6.caffemodel';
net_def = '../../v9/cifar10/net_def/train_val_v9.0_neat_deploy.prototxt';

% data
dataset = 'cifar10';
multi_crop = false;
mean_path = '/home/hyli/dataset/cifar';
% returns mean_data in W x H x C with BGR channels
mean_data = caffe.io.read_mean([mean_path '/cifar10_mean.binaryproto']);

switch dataset
    case 'cifar10'
        test_batch = load('/home/hyli/dataset/cifar_matlab/test_batch.mat');
        images = test_batch.data;
        % from 1 to 10
        labels = test_batch.labels + 1;
        
    case 'cifar100'
        test_batch = load('/home/hyli/dataset/cifar_matlab/cifar100/test.mat');
        images = test_batch.data;
        % from 1 to 100
        labels = test_batch.fine_labels + 1;
end

caffe.set_mode_gpu();
caffe.set_device(gpu_id);
% Initialize a network
net = caffe.Net(net_def, net_weights, phase);
if multi_crop
    net.blobs('data').reshape([32 32 3 10]);
else
    net.blobs('data').reshape([32 32 3 1]);
end
net.reshape();

CROPPED_DIM = 32;
cnt = 0;
for which_im = 1:length(labels)
    
    im = reshape(images(which_im, :), [32, 32, 3]);
    im = permute(im, [2, 1, 3]);
    %imshow(im);
    % preprocess
    im_ = im(:, :, [3,2,1]);
    im_ = permute(im_, [2,1,3]);
    im_ = single(im_);
    im_ = im_ - mean_data;
    
    if multi_crop
        IMAGE_DIM = randi([32, 40]);
        im_ = imresize(im_, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        % oversample (4 corners, center, and their x-axis flips)
        crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
        indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
        n = 1;
        for i = indices
            for j = indices
                crops_data(:, :, :, n) = im_(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
                crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
                n = n + 1;
            end
        end
        center = floor(indices(2) / 2) + 1;
        crops_data(:,:,:,5) = ...
            im_(center:center+CROPPED_DIM-1, center:center+CROPPED_DIM-1,:);
        crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);
        input = crops_data;
    else
        input = im_;
    end
    
    % do forward
    res = net.forward({input});
    prob = res{1};
    [~, pred] = max(mean(prob, 2));
    gt = labels(which_im);
    
    if gt == pred
        cnt = cnt + 1;
    end
    
    if ~mod(which_im, 1000)
        fprintf('processed %d images, accuracy=%.2f ...\n', ...
            which_im, cnt/which_im);
    end
end

accuracy = cnt/length(labels);


