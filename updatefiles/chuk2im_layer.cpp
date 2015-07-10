#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2chuk.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Chuk2imLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ChunkingParameter chuk_param = this->layer_param_.chunking_param();
  CHECK((!chuk_param.has_im_size() && chuk_param.has_im_h()
      && chuk_param.has_im_w())
      || (!chuk_param.has_im_h() && !chuk_param.has_im_w()))
  << "image size is im_size OR im_h and im_w; not both";
  if (!chuk_param.has_im_h()) {
    im_h_ = im_w_ = chuk_param.im_size();
  } else {
    im_h_ = chuk_param.im_h();
    im_w_ = chuk_param.im_w();
  }
  CHECK_GT(im_h_, 0) << "the height of image cannot be zero.";
  CHECK_GT(im_w_, 0) << "the width of image cannot be zero.";
}

template <typename Dtype>
void Chuk2imLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  im_c_ = channels_ / (im_h_ / height_) / (im_w_ / width_);
  top[0]->Reshape( bottom[0]->num(), im_c_, im_h_, im_w_ );
}

template <typename Dtype>
void Chuk2imLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    chuk2im_cpu(bottom_data + bottom[0]->offset(n), im_c_, im_h_, im_w_,
        height_, width_, 1, 1,
        height_, width_, top_data + top[0]->offset(n));
  }
}

template <typename Dtype>
void Chuk2imLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    im2chuk_cpu(top_diff + top[0]->offset(n), im_c_, im_h_, im_w_,
        height_, width_, 1, 1,
        height_, width_, bottom_diff + bottom[0]->offset(n));
  }
}

#ifdef CPU_ONLY
STUB_GPU(Chuk2imLayer);
#endif

INSTANTIATE_CLASS(Chuk2imLayer);
REGISTER_LAYER_CLASS(Chuk2im);

}  // namespace caffe
