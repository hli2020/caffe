# parse log
# in '~/research_office/archive_caffe_2/caffe_all_new_7_29_server/tools/extra'
sh parse_log_cifar.sh /media/hongyang/deep_learning/project_and_beyond/bias_project/bias_cifar/v8/log/cifar_v8.0_1_aug.log

# draw network structure
python draw_net.py --rankdir="TB" /media/hongyang/deep_learning/project_and_beyond/bias_project/bias_cifar/v8/train_val_v8.0.prototxt v8.0.pdf

