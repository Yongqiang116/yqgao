#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/im2chuk.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

// Forward declare kernel functions
template <typename Dtype>
__global__ void im2chuk_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, 
    const int stride_h, const int stride_w,
    const int height_chuk, const int width_chuk,
    Dtype* data_chuk);

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class Im2chukKernelTest : public ::testing::Test {
 protected:
  Im2chukKernelTest()
        // big so launches > 1024 threads
      : blob_bottom_(new Blob<Dtype>(5, 500, 10, 10)),
        blob_top_(new Blob<Dtype>()),
        blob_top_cpu_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    height_ = blob_bottom_->height();
    width_ = blob_bottom_->width();
    channels_ = blob_bottom_->channels();
    local_ = 2;
    stride_ = 2;
    kernel_size_ = 3;
    height_chuk_ = kernel_size_ + stride_ * (local_ - 1);
    width_chuk_ = kernel_size_ + stride_ * (local_ - 1);
  }

  virtual ~Im2chukKernelTest() {
      delete blob_bottom_;
      delete blob_top_;
      delete blob_top_cpu_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_cpu_;
  int height_;
  int width_;
  int channels_;
  int local_;
  int stride_;
  int kernel_size_;
  int height_chuk_;
  int width_chuk_;
};

TYPED_TEST_CASE(Im2chukKernelTest, TestDtypes);

TYPED_TEST(Im2chukKernelTest, TestGPU) {
  Caffe::set_mode(Caffe::GPU);

  // Reshape the blobs to correct size for im2col output
  this->blob_top_->Reshape(this->blob_bottom_->num(),
          this->channels_ * ((this->height_ - this->kernel_size_) / this->local_ / this->stride_ + 1) 
          * ((this->width_ - this->kernel_size_) / this->local_ / this->stride_ + 1),
          this->height_chuk_,
          this->width_chuk_);

  this->blob_top_cpu_->Reshape(this->blob_bottom_->num(),
          this->channels_ * ((this->height_ - this->kernel_size_) / this->local_ / this->stride_ + 1) 
          * ((this->width_ - this->kernel_size_) / this->local_ / this->stride_ + 1),
          this->height_chuk_,
          this->width_chuk_);

  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  TypeParam* cpu_data = this->blob_top_cpu_->mutable_cpu_data();

  // CPU Version
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    im2chuk_cpu(this->blob_bottom_->cpu_data() + this->blob_bottom_->offset(n),
      this->channels_, this->height_, this->width_,
      this->kernel_size_, this->kernel_size_, this->local_, this->local_,
      this->stride_, this->stride_,
      cpu_data + this->blob_top_cpu_->offset(n));
  }

  // GPU version
  int num_kernels = this->channels_ * ((this->height_ - this->kernel_size_) / this->local_ / this->stride_ + 1) 
  * ((this->width_ - this->kernel_size_) / this->local_ / this->stride_ + 1);
  int default_grid_dim = CAFFE_GET_BLOCKS(num_kernels);

  // Launch with different grid sizes
  for (int grid_div = 2; grid_div <= 8; grid_div++) {
    for (int n = 0; n < this->blob_bottom_->num(); ++n) {
      int grid_dim = default_grid_dim/grid_div;
      // NOLINT_NEXT_LINE(whitespace/operators)
      im2chuk_gpu_kernel<TypeParam><<<grid_dim, CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, bottom_data + this->blob_bottom_->offset(n),
        this->height_, this->width_, 
        this->kernel_size_ * this->local_, this->kernel_size_ * this->local_,
        this->height_chuk_, this->width_chuk_,
        top_data + this->blob_top_->offset(n));
      CUDA_POST_KERNEL_CHECK;
    }

    // Compare results against CPU version
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      TypeParam cpuval = cpu_data[i];
      TypeParam gpuval = this->blob_top_->cpu_data()[i];
      EXPECT_EQ(cpuval, gpuval);
      if (cpuval != gpuval) {
        break;
      }
    }
  }
}

}  // namespace caffe
