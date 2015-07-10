#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class Im2chukLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  Im2chukLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 9, 8)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~Im2chukLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(Im2chukLayerTest, TestDtypesAndDevices);

TYPED_TEST(Im2chukLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ChunkingParameter* chunking_param =
      layer_param.mutable_chunking_param();
  chunking_param->set_kernel_size(3);
  chunking_param->set_stride(2);
  chunking_param->set_local(2);
  Im2chukLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 6);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(Im2chukLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ChunkingParameter* chunking_param =
      layer_param.mutable_chunking_param();
  chunking_param->set_kernel_size(3);
  chunking_param->set_stride(2);
  chunking_param->set_local(2);
  Im2chukLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // We are lazy and will only check the top left block
  for (int c = 0; c < 6; ++c) {
    EXPECT_EQ(this->blob_bottom_->data_at(0, (c % 3), (c / 3 / 1) % 2 * 4, (c / 3) % 1 * 4),
        this->blob_top_->data_at(0, c, 0, 0));
  }
}

TYPED_TEST(Im2chukLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ChunkingParameter* chunking_param =
      layer_param.mutable_chunking_param();
  chunking_param->set_kernel_size(3);
  chunking_param->set_stride(2);
  chunking_param->set_local(2);
  Im2chukLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(Im2chukLayerTest, TestRect) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ChunkingParameter* chunking_param =
      layer_param.mutable_chunking_param();
  chunking_param->set_kernel_h(3);
  chunking_param->set_kernel_w(2);
  chunking_param->set_stride(2);
  chunking_param->set_local(2);
  Im2chukLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // We are lazy and will only check the top left block
  for (int c = 0; c < 45; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, (c % 3), (c / 3 / 2) % 2 * 4, (c / 3) % 2 * 4));
  }
}


TYPED_TEST(Im2chukLayerTest, TestRectGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ChunkingParameter* chunking_param =
      layer_param.mutable_chunking_param();
  chunking_param->set_kernel_h(5);
  chunking_param->set_kernel_w(3);
  chunking_param->set_stride(2);
  chunking_param->set_local(2);
  Im2chukLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
