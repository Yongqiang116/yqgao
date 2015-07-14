cp test_im2chuk.cpp test_im2chuk_kernel.cu ../src/caffe/test
cp im2chuk.cpp im2chuk.cu ../src/caffe/util
cp vision_layers.hpp ../include/caffe
cp im2chuk_layer.cpp im2chuk_layer.cu chuk2im_layer.cpp chuk2im_layer.cu ../src/caffe/layers
cp caffe.proto ../src/caffe/proto
cp im2chuk.hpp ../include/caffe/util
cp eltwise_layer.cpp eltwise_layer.cu ../src/caffe/layers
