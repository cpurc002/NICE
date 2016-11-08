// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CPP_INCLUDE_GPU_KMEANS_H_
#define CPP_INCLUDE_GPU_KMEANS_H_
#ifdef NEED_CUDA

#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <Eigen/Dense>
#include <unistd.h>
#include <stdexcept>
#include <ctime>

#include <iostream>

#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "include/gpu_svd_solver.h"
#include "include/util.h"
#include "include/kmeans_kernels.h"
#include "include/kmcuda.h"

namespace Nice {

// Abstract class of common matrix operation interface
template <typename T>
class GpuKmeans {
 private:
  int num_clusters_;
  int num_iterations_;
  int num_points_;
  Matrix<T> *data_points_;
  Vector<unsigned int> *labels_;
  Matrix<T> *centers_;
  bool kmpp_;
  float tolerance_;
  float yinyang_t_;
  uint32_t seed_;
  uint32_t device_;
  int32_t  verbosity_;


  void doKmeans(const Matrix<T> &input_data) {
    Timer timer;

//  setDevice();
    startTime(&timer);
    std::cout << "Allocating Memories" << std::endl;
    int numK = num_clusters_;
    Matrix<float> casted_samples = input_data.template cast<float>();
    Matrix<float> clusters(numK, input_data.cols());
    Matrix<T> casted_clusters(numK, input_data.cols());
    Vector<unsigned int> labels(input_data.rows());
    bool kmpp = kmpp_;
    float tolerance = tolerance_;
    float yinyang_t = yinyang_t_;
    uint32_t samples_size = input_data.rows();
    uint16_t features_size = input_data.cols();
    uint32_t clusters_size = num_clusters_;
    uint32_t seed = seed_;
    uint32_t device = device_;
    int32_t  verbosity = verbosity_;
    float *samples = &casted_samples(0);
    float *centroids = &clusters(0);
    uint32_t *assignments = &labels(0);
    std::cout << "Finished allocating memories" << std::endl;
    std::cout << "  Elapsed time to allocate memories ";
    std::cout << elapsedTime(timer) << std::endl;


    startTime(&timer);
    std::cout << "\nRunning Kmeans" << std::endl;
    kmeans_cuda(kmpp, tolerance, yinyang_t, samples_size,
                    features_size, clusters_size, seed,
                    device, verbosity, samples,
                    centroids, assignments);
    std::cout << "Finished running Kmeans" << std::endl;
    std::cout << "  Elapsed time to run Kmeans: ";
    std::cout << elapsedTime(timer) << std::endl;

    num_points_ = input_data.rows();
    num_clusters_ = numK;
    casted_clusters = clusters.template cast<T>();
    centers_ = &casted_clusters;
    labels_ = &labels;
  }

 public:
  GpuKmeans() {
    kmpp_ = true;
    tolerance_ = 0.001;
    yinyang_t_ = 0.1;
    seed_ = time(NULL);
    device_ = 0;
    verbosity_ = 1;
    num_clusters_ = 5;
    num_iterations_ = 500;
    num_points_ = 5000;
    data_points_ = NULL;
    labels_ = NULL;
    centers_ = NULL;
  }

  void setKmpp(bool kmpp) { kmpp_ = kmpp; }
  void setTolerance(float tolerance) { tolerance_ = tolerance; }
  void setYinYang(float yinyang_t) { yinyang_t_ = yinyang_t; }
  void setSeed(uint32_t seed) { seed_ = seed; }
  void setDevice(uint32_t device) { device_ = device; }
  void setVerbosity(int32_t verbosity) { verbosity_ = verbosity; }
  void setNumClusters(int num_clusters) { num_clusters_ = num_clusters; }
  void setNumIterations(int num_iterations) {num_iterations_ = num_iterations;}
  void setNumPoints(int num_points) { num_points_ = num_points; }
  void setDataPoints(Matrix<T> data_points) { data_points_ = &data_points; }
  void setLabels(Vector<T> labels) { labels_ = labels; }
  void setCenters(Matrix<T> centers) { centers_ = centers; }

  Matrix<T>* Fit(const Matrix<T> &input_data) {
    doKmeans(input_data);
    return centers_;
  }
  Vector<unsigned int>* Predict(const Matrix<T> &new_points) {
    int numObjs = new_points.rows();
    int numDims = new_points.cols();
    int numK = num_clusters_;
//    Matrix<T> * clusters = centers_;
    Vector<unsigned int> * new_labels;
    new_labels = new Vector<unsigned int>(numObjs);
    unsigned int closestClusterID = 0;
    unsigned int minDistance = -1;
    float dimDist = 0;
    float multiDimDist = 0;
    for (int i = 0; i < numObjs; ++i) {
      for (int j = 0; j < numK; ++j) {
        for (int k = 0; k < numDims; ++k) {
          dimDist = new_points(i, k) - (*centers_)(j, k);
          multiDimDist += dimDist * dimDist;
        }
        if (multiDimDist < minDistance) {
          closestClusterID = (*labels_)(j);
        }
      }
      minDistance = -1;
      multiDimDist = 0;
      (*new_labels)(i) = closestClusterID;
    }
    return new_labels;
  }
  Vector<T> *GetLabels() { return labels_; }
};

}  // namespace Nice
#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_GPU_KMEANS_H_
