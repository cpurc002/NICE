#include <stdint.h>
#include<cuda_runtime.h>                                                        
#include<device_launch_parameters.h>                                            
#include<cuda_runtime_api.h>   

#include <iostream>
 
#include "include/kmeans_kernels.h"
#include "include/gpu_util.h"

/*__global__ void copyMemory(float * dst, int numDims) { 
   
  unsigned int idx = threadIdx.x; 
  for (int i = 0; i < numDims; ++i) {
    dst[idx*numDims + i] = c_clusters[idx * numDims + i]; 
  }
}*/
namespace Nice { 

template <typename T>
__global__ void classifyPoints(T *data, T *clusters, 
                              T *distances, int *labels, 
                              int numK, int numObjs, int numDims) {
  unsigned int clusterID = blockIdx.x;  // Each K has own Block 
  unsigned int dataID = threadIdx.x;  // Each sample has own Thread
  unsigned int distanceID = threadIdx.x + blockIdx.x * numObjs;  // #Threads x #Blocks 
  unsigned int labelID = threadIdx.x;            // data point id
  unsigned int numThreads = blockDim.x; 

  T dimDist = 0;  
  T dataDimVal = 0; 
  T clusterDimVal = 0; 
  T minDistance = 0;
  int closestClusterID = 0;  
  int multiDimDist = 0;  
  int stride = 0; 

  for (int l = 0; l < (numObjs + numThreads - 1) / numThreads; ++l) {  
    stride = numThreads * l; 
    if (dataID + stride < numObjs) { 
      multiDimDist = 0; 
      for (int i = 0; i < numDims; ++i) { 
        clusterDimVal = *(clusters + clusterID + i * numK);
        dataDimVal = *(data + dataID + i * numObjs + stride); 
        dimDist = (dataDimVal - clusterDimVal) * (dataDimVal - clusterDimVal);  
        multiDimDist += dimDist;
      }
      *(distances + distanceID + stride) = multiDimDist; 
    
      __syncthreads(); 
    
      if (blockIdx.x == 0) { 
        minDistance = *(distances + distanceID + stride); 
        closestClusterID = 0;  
        for (int i = 1; i < numK; ++i) { 
          if (*(distances + distanceID + i * numObjs + stride) < minDistance) {  
            minDistance = *(distances + distanceID + i * numObjs + stride);
            closestClusterID = i;
          }
        }
       *(labels + labelID + stride) = closestClusterID; 
       }
     }
     __syncthreads();
   }
}

template <typename T>
void doKmeans(cudaDeviceProp deviceProp, T *data, T *clusters, int *labels,
              T *distances, int numK, int numObjs, int numDims) { 

  int MAX_THREADS_BLOCK = deviceProp.maxThreadsPerBlock; 
  int MAX_SHARED_MEMORY = deviceProp.sharedMemPerBlock; 
  int MAX_THREADS_GRID = 2048 * deviceProp.multiProcessorCount; 
  int NEEDED_THREADS_BLOCK = numObjs; 
  int NEEDED_SHARED_MEMORY = numK * numDims * sizeof(T); 
  int THREADS_BLOCK;   
  int BLOCKS_GRID = numK; 

  if (NEEDED_SHARED_MEMORY > MAX_SHARED_MEMORY) { 
    std::cout<<"Not enough shared"<<std::endl; 
  }
  
  if (NEEDED_THREADS_BLOCK > MAX_THREADS_BLOCK) { 
    THREADS_BLOCK = MAX_THREADS_BLOCK; 
    std::cout<<"Too many samples for 1:1 threads"<<std::endl; 
  }
  else {
    THREADS_BLOCK = NEEDED_THREADS_BLOCK; 
  } 

  if (THREADS_BLOCK * BLOCKS_GRID > MAX_THREADS_GRID) { 
//    THREADS_BLOCK = MAX_THREADS_GRID / BLOCKS_GRID; 
  }
  std::cout<<"Number of threads set to : "<<THREADS_BLOCK<<std::endl; 
  std::cout<<"Number of blocks set to : "<<BLOCKS_GRID<<std::endl; 
  
  

  dim3 grid_dim, block_dim;
  grid_dim.x = BLOCKS_GRID;
  block_dim.x = THREADS_BLOCK;

  grid_dim.y = grid_dim.z = 1;
  block_dim.y = block_dim.z = 1;

  classifyPoints<<<grid_dim, block_dim>>>(data, clusters, distances, 
                                          labels, numK, numObjs, numDims);
} 
template __global__ void classifyPoints<float>(float *data, float *clusters,                            
                               float *distances, int *labels,                   
                               int numK, int numObjs, int numDims);
template __global__ void classifyPoints<double> (double *data, double *clusters,                            
                               double *distances, int *labels,                   
                               int numK, int numObjs, int numDims);

template void doKmeans<float>(cudaDeviceProp DeviceProp, float *data, float *clusters, int *labels,     
              float *distances, int numK, int numObjs, int numDims) ;
template void doKmeans<double> (cudaDeviceProp DeviceProp, double *data, double *clusters, int *labels,     
              double *distances, int numK, int numObjs, int numDims) ;

}
