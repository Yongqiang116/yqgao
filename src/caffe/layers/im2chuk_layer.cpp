#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2chuk.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Im2chukLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ChunkingParameter chuk_param = this->layer_param_.chunking_param();
  CHECK(!chuk_param.has_kernel_size() !=
      !(chuk_param.has_kernel_h() && chuk_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(chuk_param.has_kernel_size() ||
      (chuk_param.has_kernel_h() && chuk_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!chuk_param.has_local_size() && chuk_param.has_local_h()
      && chuk_param.has_local_w())
      || (!chuk_param.has_local_h() && !chuk_param.has_local_w()))
      << "local is local OR local_h and local_w are required.";
  CHECK((!chuk_param.has_stride() && chuk_param.has_stride_h()
      && chuk_param.has_stride_w())
      || (!chuk_param.has_stride_h() && !chuk_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (chuk_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = chuk_param.kernel_size();
  } else {
    kernel_h_ = chuk_param.kernel_h();
    kernel_w_ = chuk_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!chuk_param.has_local_h()) {
    local_h_ = local_w_ = chuk_param.local_size();
  } else {
    local_h_ = chuk_param.local_h();
    local_w_ = chuk_param.local_w();
  }
  if (!chuk_param.has_stride_h()) {
    stride_h_ = stride_w_ = chuk_param.stride();
  } else {
    stride_h_ = chuk_param.stride_h();
    stride_w_ = chuk_param.stride_w();
  }
}

template <typename Dtype>
void Im2chukLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // compute some paramaters
  int chuk_h_ = kernel_h_ + stride_h_ * (local_h_ - 1);
  int chuk_w_ = kernel_w_ + stride_w_ * (local_w_ - 1);

  top[0]->Reshape(
      bottom[0]->num(), 
      channels_ * ((height_ - chuk_h_) / stride_h_ / local_h_ + 1) * 
                  ((width_ - chuk_w_) / stride_w_ / local_w_ + 1),
      chuk_h_, chuk_w_);
}

template <typename Dtype>
void Im2chukLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    im2chuk_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
        width_, kernel_h_, kernel_w_, local_h_, local_w_,
        stride_h_, stride_w_, top_data + top[0]->offset(n));
  }
}

template <typename Dtype>
void Im2chukLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    chuk2im_cpu(top_diff + top[0]->offset(n), channels_, height_, width_,
        kernel_h_, kernel_w_, local_h_, local_w_,
        stride_h_, stride_w_, bottom_diff + bottom[0]->offset(n));
  }
}

#ifdef CPU_ONLY
STUB_GPU(Im2chukLayer);
#endif

INSTANTIATE_CLASS(Im2chukLayer);
REGISTER_LAYER_CLASS(Im2chuk);

}  // namespace caffe
