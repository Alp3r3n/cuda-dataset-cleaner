#include "cuda_buffer.hpp"
#include <cuda_runtime.h>
#include <cstdio>

CudaBuffer::CudaBuffer() : ptr_(nullptr), size_(0) {}

CudaBuffer::CudaBuffer(size_t bytes) : ptr_(nullptr), size_(0) {
    allocate(bytes);
}

CudaBuffer::~CudaBuffer() {
    free();
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_  = nullptr;
    other.size_ = 0;
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
        free();
        ptr_       = other.ptr_;
        size_      = other.size_;
        other.ptr_  = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void CudaBuffer::allocate(size_t bytes) {
    free();
    if (bytes == 0) return;
    cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "CudaBuffer::allocate(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        ptr_  = nullptr;
        size_ = 0;
        return;
    }
    size_ = bytes;
}

void CudaBuffer::ensureCapacity(size_t bytes) {
    if (ptr_ != nullptr && bytes <= size_) return;
    allocate(bytes);
}

void CudaBuffer::free() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
    size_ = 0;
}

void*       CudaBuffer::data()       { return ptr_; }
const void* CudaBuffer::data() const { return ptr_; }
size_t      CudaBuffer::size() const { return size_; }
