#include <stdint.h>
#include<cuda_runtime.h>                                                        
#include<device_launch_parameters.h>                                            
#include<cuda_runtime_api.h>   

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

template <typename T>
void doKmeans(cudaDeviceProp DeviceProp, T *data, T *clusters, int *labels,
              float *distances, int numK, int numObjs, int numDims) { 

  dim3 grid_dim, block_dim;
  grid_dim.x = numObjs;
  block_dim.x = numK;

  grid_dim.y = grid_dim.z = 1;
  block_dim.y = block_dim.z = 1;

  classifyPoints<<<grid_dim, block_dim>>>(data, clusters, distances, 
                                          labels, numK, numDims);

   cudaFree(distances); 
}

template __global__ void classifyPoints<float>(float *data, float *clusters,                            
                               float *distances, int *labels,                   
                               int numK, int numDims);
template __global__ void classifyPoints<double> (double *data, double *clusters,                            
                               float *distances, int *labels,                   
                               int numK, int numDims);

template void doKmeans<float>(cudaDeviceProp DeviceProp, float *data, float *clusters, int *labels,     
              float *distances, int numK, int numObjs, int numDims) ;
template void doKmeans<double> (cudaDeviceProp DeviceProp, double *data, double *clusters, int *labels,     
              float *distances, int numK, int numObjs, int numDims) ;

}