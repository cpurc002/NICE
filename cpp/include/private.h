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
#ifndef CPP_INCLUDE_PRIVATE_H_
#define CPP_INCLUDE_PRIVATE_H_

#include "include/kmcuda.h"

enum KMCUDAInitMethod {
  kmcudaInitMethodRandom = 0,
  kmcudaInitMethodPlusPlus
};

#define RETERR(call, ...) do { \
  auto __r = call; \
  if (__r != kmcudaSuccess) { \
    __VA_ARGS__; \
    return __r; \
  } \
} while (false)

#define INFO(...) do { if (verbosity > 0) {printf(__VA_ARGS__);}} while (false)
#define DEBUG(...) do { if (verbosity > 1) {printf(__VA_ARGS__);}} while (false)

extern "C" {

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t samples_size, uint32_t cc, float *samples, float *centroids,
    float *dists, float *distssum, float **dev_sums);

KMCUDAResult kmeans_cuda_setup(uint32_t samples_size, uint16_t features_size,
                               uint32_t clusters_size, uint32_t yy_groups_size,
                               uint32_t device, int32_t verbosity);

KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t yinyang_groups, uint32_t samples_size_,
    uint32_t clusters_size_, uint16_t features_size, int32_t verbosity,
    const float *samples, float *centroids, uint32_t *ccounts,
    uint32_t *assignments_prev, uint32_t *assignments, uint32_t *assignments_yy,
    float *centroids_yy, float *bounds_yy, float *drifts_yy,
    uint32_t *passed_yy);

KMCUDAResult kmeans_init_centroids(
    KMCUDAInitMethod method, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, int32_t verbosity, float *samples,
    void *dists, float *centroids);
}

#endif  // CPP_INCLUDE_PRIVATE_H_
