#pragma once
#include <cstddef>

// RAII wrapper for a CUDA device-memory allocation.
// Non-copyable, movable. ensureCapacity() allows reuse across iterations.
class CudaBuffer {
public:
    CudaBuffer();
    explicit CudaBuffer(size_t bytes);
    ~CudaBuffer();

    CudaBuffer(const CudaBuffer&)            = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    // Replace the existing allocation with one of `bytes` bytes.
    // Frees the old allocation first. Returns true on success.
    bool allocate(size_t bytes);

    // Grow the allocation to at least `bytes` bytes. No-op if already large enough.
    // Returns true on success (including the no-op case).
    bool ensureCapacity(size_t bytes);

    // Release the current allocation (if any). Safe to call multiple times.
    void free();

    void*       data();
    const void* data() const;
    size_t      size() const;

private:
    void*  ptr_;
    size_t size_;
};
