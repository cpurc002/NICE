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
#ifndef CPP_INCLUDE_KERNELS_H_
#define CPP_INCLUDE_KERNELS_H_

#ifdef NEED_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cusolverDn.h>
#include <ctime>
#include <time.h>
#include <iostream>
#include <sys/time.h>                                                           
#include <string>                                                               
#include <stdint.h>                                                             

namespace Nice {

template<typename T>                                                            
__global__ void classifyPoints(T *data, T *clusters,                            
                               float *distances, int *labels,                   
                               int numK, int numDims) {                         
  unsigned int clusterID = threadIdx.x * numDims;  // Each Thread has its own K 
  unsigned int dataID = blockIdx.x * numDims;  // data point id * numDims       
  unsigned int distanceID = blockIdx.x * numK;  // data point id * numK         
  unsigned int labelID = blockIdx.x;            // data point id                
  float multiDimDist = 0;                                                       
  float dimDist = 0;                                                            
  float dataDimVal = 0;                                                         
  float clusterDimVal = 0;                                                      
  float minDistance = 0;                                                        
  int closestClusterID = 0;                                                     
                                                                                
  for (int j = 0; j < numDims; ++j) {                                           
      dataDimVal = *(data + dataID + j);                                        
      clusterDimVal = *(clusters + clusterID + j);                              
      dimDist = (dataDimVal - clusterDimVal) * (dataDimVal - clusterDimVal);    
      multiDimDist += dimDist;                                                  
  }                                                                             
  *(distances + distanceID + threadIdx.x) = multiDimDist;                       
                                                                                
  __syncthreads();                                                              
                                                                                
  if (threadIdx.x == 0) {                                                       
    minDistance = *(distances + distanceID);                                    
    closestClusterID = 0;                                                       
    for (int i = 1; i < numK; ++i) {                                            
      if (*(distances + distanceID + i) < minDistance) {                        
        minDistance = *(distances + distanceID + i);                            
        closestClusterID = i;                                                   
      }                                                                         
    }                                                                           
   *(labels + labelID) = closestClusterID;                                      
   }                                                                            
}
template<typename T>                                                            
void doKmeans(cudaDeviceProp DeviceProp, T *data, T *clusters, int *labels,     
              float *distances, int numK, int numObjs, int numDims) {           
                                                                                
  dim3 grid_dim, block_dim;                                                     
  grid_dim.x = numObjs;                                                         
  block_dim.x = numK;                                                           
                                                                                
  grid_dim.y = grid_dim.z = 1;                                                  
  block_dim.y = block_dim.z = 1;                                                
                                                                                
  classifyPoints<T><<<grid_dim, block_dim>>>(data, clusters, distances,         
                                          labels, numK, numDims);               
                                                                                
   cudaFree(distances);                                                         
} 

/*
template void doKmeans<float>(cudaDeviceProp DeviceProp, float *data, float *clusters, int *labels,     
              float *distances, int numK, int numObjs, int numDims) ;
template void doKmeans<double> (cudaDeviceProp DeviceProp, double *data, double *clusters, int *labels,     
              float *distances, int numK, int numObjs, int numDims) ;
template __global__ void classifyPoints<float>(float *data, float *clusters,                            
                               float *distances, int *labels,                   
                               int numK, int numDims);
template __global__ void classifyPoints<double> (double *data, double *clusters,                            
                               float *distances, int *labels,                   
                               int numK, int numDims);

*/
}  // namespace Nice

#endif  // NEED_CUDA
#endif  // CPP_INCLUDE_KERNELS_H_
