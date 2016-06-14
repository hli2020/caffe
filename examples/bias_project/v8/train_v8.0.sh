#!/usr/bin/env sh

#  	--weights=../benchmark/result/cifar10_quick_iter_7000.caffemodel \
CAFFE_PATH=/home/hongyang/research_office/caffe_history/build/tools

GLOG_logtostderr=1 $CAFFE_PATH/caffe train \
  		--solver=solver_v8.0_long_lr.prototxt \
  		--gpu=0 \
2>&1 | tee log/cifar_v8.0_1_aug_c100_small_rescale.log 


  

