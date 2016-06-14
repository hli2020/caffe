#!/usr/bin/env sh

#  	--weights=../benchmark/result/cifar10_quick_iter_7000.caffemodel \
CAFFE_PATH=/home/hongyang/research_office/archive_caffe_2/caffe_all_new_7_29_server/build/tools

GLOG_logtostderr=1 $CAFFE_PATH/caffe train \
  		--solver=solver_v8.0.prototxt \
  		--gpu=0 \
2>&1 | tee log/cifar_v8.0_1.log 


  

