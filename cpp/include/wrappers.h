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
#ifndef CPP_INCLUDE_WRAPPERS_H_
#define CPP_INCLUDE_WRAPPERS_H_

#include <cuda_runtime_api.h>
#include <memory>

using unique_devptr_parent = std::unique_ptr<void, std::function<void(void*)>>;

class unique_devptr : public unique_devptr_parent {
 public:
  explicit unique_devptr(void *ptr) : unique_devptr_parent(
      ptr, [](void *p){ cudaFree(p); }) {}
};

using unique_devptrptr_parent =
                std::unique_ptr<void*, std::function<void(void**)>>;

class unique_devptrptr : public unique_devptrptr_parent {
 public:
  explicit unique_devptrptr(void **ptr) : unique_devptrptr_parent(
      ptr, [](void **p){ if (p) { cudaFree(*p); } }) {}
};

#endif  // CPP_INCLUDE_WRAPPERS_H_
