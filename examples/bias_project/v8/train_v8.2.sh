#!/usr/bin/env sh

#  	--weights=../benchmark/result/cifar10_quick_iter_7000.caffemodel \
CAFFE_PATH=/home/hongyang/research_office/caffe_history/build/tools

GLOG_logtostderr=1 $CAFFE_PATH/caffe train \
  		--solver=solver_v8.2.prototxt \
  		--gpu=1 \
2>&1 | tee log/cifar_v8.2_wider_dp_conv.log 


  

