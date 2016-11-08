// The MIT License (MIT)
//
// Copyright (c) 2016 source{d}.
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

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <memory>
#include "include/kmcuda.h"

static char module_docstring[] =
    "This module provides fast K-means implementation which uses CUDA.";
static char kmeans_cuda_docstring[] =
    "Assigns cluster label to each sample and calculates cluster centers.";

static PyObject *py_kmeans_cuda(PyObject *self, PyObject *args,
                                PyObject *kwargs);

static PyMethodDef module_functions[] = {
  {"kmeans_cuda", reinterpret_cast<PyCFunction>(py_kmeans_cuda),
  METH_VARARGS | METH_KEYWORDS, kmeans_cuda_docstring},
  {NULL, NULL, 0, NULL}
};

extern "C" {
PyMODINIT_FUNC PyInit_libKMCUDA(void) {
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "libKMCUDA",         /* m_name */
      module_docstring,    /* m_doc */
      -1,                  /* m_size */
      module_functions,    /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
  };
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "PyModule_Create() failed");
    return NULL;
  }
  // numpy
  import_array();
  return m;
}
}

using pyobj_parent = std::unique_ptr<PyObject, std::function<void(PyObject*)>>;

class pyobj : public pyobj_parent {
 public:
  explicit pyobj(PyObject *ptr) : pyobj_parent(
      ptr, [](PyObject *p){ Py_DECREF(p); }) {}
};

static PyObject *py_kmeans_cuda(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  uint32_t clusters_size = 0, seed = static_cast<uint32_t>(time(NULL));
  uint32_t device = 0;
  int32_t verbosity = 0;
  float tolerance = .0, yinyang_t = .1;
  PyObject *kmpp = Py_False;
  PyObject *samples_obj;
  static const char *kwlist[] = {"samples", "clusters", "tolerance", "kmpp",
                              "yinyang_t", "seed", "device", "verbosity", NULL};

  /* Parse the input tuple */
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "OI|fO!fIIi", const_cast<char**>(kwlist),
      &samples_obj, &clusters_size, &tolerance, &PyBool_Type, &kmpp, &yinyang_t,
      &seed, &device, &verbosity)) {
    return NULL;
  }
  if (clusters_size < 2 || clusters_size == UINT32_MAX) {
    PyErr_SetString(PyExc_ValueError, "\"clusters\" must be greater than 1 and "
                                      "less than (1 << 32) - 1");
    return NULL;
  }
  pyobj samples_array(PyArray_FROM_OTF(samples_obj,
                      NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  if (samples_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "\"samples\" must be a 2D numpy array");
    return NULL;
  }
  auto ndims = PyArray_NDIM(
                         reinterpret_cast<PyArrayObject*>(samples_array.get()));
  if (ndims != 2) {
    PyErr_SetString(PyExc_ValueError, "\"samples\" must be a 2D numpy array");
    return NULL;
  }
  auto dims = PyArray_DIMS(
                         reinterpret_cast<PyArrayObject*>(samples_array.get()));
  uint32_t samples_size = static_cast<uint32_t>(dims[0]);
  uint32_t features_size = static_cast<uint32_t>(dims[1]);
  if (features_size > UINT16_MAX) {
    char msg[128];
    snprintf(msg, "\"samples\": more than %" PRIu32 " features not supported",
            features_size);
    PyErr_SetString(PyExc_ValueError, msg);
    return NULL;
  }
  float *samples = reinterpret_cast<float*>(PyArray_DATA(
      reinterpret_cast<PyArrayObject*>(samples_array.get())));
  npy_intp centroid_dims[] = {clusters_size, features_size, 0};
  auto centroids_array = PyArray_EMPTY(2, centroid_dims, NPY_FLOAT32, false);
  float *centroids = reinterpret_cast<float*>(PyArray_DATA(
      reinterpret_cast<PyArrayObject*>(centroids_array)));
  npy_intp assignments_dims[] = {samples_size, 0};
  auto assignments_array = PyArray_EMPTY(1, assignments_dims,
                                         NPY_UINT32, false);
  uint32_t *assignments = reinterpret_cast<uint32_t*>(PyArray_DATA(
      reinterpret_cast<PyArrayObject*>(assignments_array)));

  int result;
  Py_BEGIN_ALLOW_THREADS
  result = kmeans_cuda(
      kmpp == Py_True, tolerance, yinyang_t, samples_size,
      static_cast<uint16_t>(features_size), clusters_size, seed, device,
      verbosity, samples, centroids, assignments);
  Py_END_ALLOW_THREADS

  switch (result) {
    case kmcudaInvalidArguments:
      PyErr_SetString(PyExc_ValueError,
                      "Invalid arguments were passed to kmeans_cuda");
      return NULL;
    case kmcudaNoSuchDevice:
      PyErr_SetString(PyExc_ValueError, "No such CUDA device exists");
      return NULL;
    case kmcudaMemoryAllocationFailure:
      PyErr_SetString(PyExc_MemoryError,
                      "Failed to allocate memory on GPU");
      return NULL;
    case kmcudaMemoryCopyError:
      PyErr_SetString(PyExc_RuntimeError, "cudaMemcpy failed");
      return NULL;
    case kmcudaRuntimeError:
      PyErr_SetString(PyExc_AssertionError, "kmeans_cuda failure (bug?)");
      return NULL;
    case kmcudaSuccess:
      return Py_BuildValue("OO", centroids_array, assignments_array);
    default:
      PyErr_SetString(PyExc_AssertionError,
                      "Unknown error code returned from kmeans_cuda");
      return NULL;
  }
}
