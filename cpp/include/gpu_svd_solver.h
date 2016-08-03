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

#ifndef CPP_INCLUDE_GPU_SVD_SOLVER_H_
#define CPP_INCLUDE_GPU_SVD_SOLVER_H_

#ifdef NEED_CUDA
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cusolverDn.h>

#include<iostream>

#include "Eigen/Dense"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"

namespace Nice {
///  This is a template class to calculate the SVD of a matrix
template<typename T>
class GpuSvdSolver {
 private:
  Matrix<T> u_;  ///< Private member variable for U matrix
  Matrix<T> v_;  ///< Private member variable for V matrix
  Vector<T> s_;  ///< Private member variable for Singular Values vector

 public:
  GpuSvdSolver() {}

/// void Compute(const Matrix<T> &a)
///      is the templated function to calculate the SVD
///
/// \param a
///  const Matrix<T> &a : Input m x n matrix to have SVD calculated
///
/// \return
/// This function outputs U and V as matrices and the Singular Values as a
/// vector and stores them in private member variables for this class u_,
/// v_, and s_ respectively.
  void Compute(const Matrix<T> &a) {
    int m = a.rows();
    int n = a.cols();
    const T *h_a = &a(0);
    // --- Setting the device matrix and moving the host matrix to the device
    T *d_a;   gpuErrchk(cudaMalloc(&d_a,      m * n * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_a, h_a, m * n * sizeof(T), cudaMemcpyHostToDevice));

    //--- host side SVD results space
    s_.resize(m, 1);
    u_.resize(m, m);
    v_.resize(n, n);

    // --- device side SVD workspace and matrices
    int work_size = 0;
    int devInfo_h = 0;
    int *devInfo;   gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));
    T *d_u;         gpuErrchk(cudaMalloc(&d_u,      m * m * sizeof(T)));
    T *d_v;         gpuErrchk(cudaMalloc(&d_v,      n * n * sizeof(T)));
    T *d_s;         gpuErrchk(cudaMalloc(&d_s,      n *     sizeof(T)));
    cusolverStatus_t stat;

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    stat = cusolverDnSgesvd_bufferSize(solver_handle, m, n, &work_size);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
      std::cout << "Initialization of cuSolver failed." << std::endl;
      cudaFree(d_s); cudaFree(d_u); cudaFree(d_v);
      cusolverDnDestroy(solver_handle);
      exit(1);
    }
    T *work;    gpuErrchk(cudaMalloc(&work, work_size * sizeof(T)));

    // --- CUDA SVD execution
    stat = GpuSvd(solver_handle, m, n,
                 d_a, d_s, d_u, d_v,
                 work, work_size, devInfo);

    // Error Check
    gpuErrchk(cudaMemcpy(&devInfo_h, devInfo,
              sizeof(int), cudaMemcpyDeviceToHost));
    if (stat != CUSOLVER_STATUS_SUCCESS || devInfo_h != 0) {
      std::cerr << "GPU SVD Solver Internal Failure" << std::endl;
      cudaFree(d_s); cudaFree(d_u); cudaFree(d_v); cudaFree(work);
      cusolverDnDestroy(solver_handle);
      exit(1);
    }
    cudaDeviceSynchronize();

        // --- Moving the results from device to host
    gpuErrchk(cudaMemcpy(&s_(0, 0), d_s, n *     sizeof(T),
              cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&u_(0, 0), d_u, m * m * sizeof(T),
              cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&v_(0, 0), d_v, n * n * sizeof(T),
              cudaMemcpyDeviceToHost));

    cudaFree(d_s); cudaFree(d_u); cudaFree(d_v); cudaFree(work);
    cusolverDnDestroy(solver_handle);
  }
  
  /// Return the matrix U after SVD decomposition                               
  ///                                                                           
  /// \param                                                                    
  /// Void                                                                      
  ///                                                                           
  /// \return                                                                   
  /// Matrix U
  Matrix<T> MatrixU() const              { return u_; }

  /// Return the matrix V after SVD decomposition                               
  ///                                                                           
  /// \param                                                                    
  /// Void                                                                      
  ///                                                                           
  /// \return                                                                   
  /// Matrix V
  Matrix<T> MatrixV() const              { return v_; }

  /// Return the Singular Values Vector after SVD decomposition                               
  ///                                                                           
  /// \param                                                                    
  /// Void                                                                      
  ///                                                                           
  /// \return                                                                   
  /// Vector Singular Values
  Vector<T> SingularValues() const       { return s_; }
};
}  // namespace Nice

#endif  // NEED_CUDA

#endif  // CPP_INCLUDE_GPU_SVD_SOLVER_H_

