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
#ifndef CPP_INCLUDE_KMCUDA_H_
#define CPP_INCLUDE_KMCUDA_H_

#include <stdint.h>

enum KMCUDAResult {
  kmcudaSuccess = 0,
  kmcudaInvalidArguments,
  kmcudaNoSuchDevice,
  kmcudaMemoryAllocationFailure,
  kmcudaRuntimeError,
  kmcudaMemoryCopyError
};

extern "C" {
/// @brief Performs K-means clustering on GPU / CUDA.
/// @param kmpp indicates whether to do kmeans++ initialization. If false,
///             ordinary random centroids will be picked.
/// @param tolerance if the number of reassignments drop below this ratio, stop.
/// @param yinyang_t the relative number of cluster groups, usually 0.1.
/// @param samples_size number of samples.
/// @param features_size number of features.
/// @param clusters_size number of clusters.
/// @param seed random generator seed passed to srand().
/// @param device CUDA device index - usually 0.
/// @param verbosity 0 - no output; 1 - progress output; >=2 - debug output.
/// @param samples input array of size samples_size x features_size in row major
///        format.
/// @param centroids output array of centroids of size clusters_size x
///        features_size in row major format.
/// @param assignments output array of cluster indices for each sample of size
///                    samples_size x 1.
/// @return KMCUDAResult.
int kmeans_cuda(bool kmpp, float tolerance, float yinyang_t,
                uint32_t samples_size,
                uint16_t features_size, uint32_t clusters_size, uint32_t seed,
                uint32_t device, int32_t verbosity, const float *samples,
                float *centroids, uint32_t *assignments);
}

#endif  // CPP_INCLUDE_KMCUDA_H_
