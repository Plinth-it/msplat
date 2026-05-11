#ifndef METAL_TENSOR_H
#define METAL_TENSOR_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <utility>

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#endif

// Forward-declare the Metal buffer type for C++ compatibility.
// Full Metal/Metal.h is only needed in .mm files.
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* MTLBufferRef;  // opaque handle in pure C++
#endif

enum class DType : uint8_t {
    Float32,
    Float16,
    Int32,
    Int64,
    UInt8,
    UInt32,
    UInt64,
    Float64,
};

inline size_t dtypeSize(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::UInt8:   return 1;
        case DType::UInt32:  return 4;
        case DType::UInt64:  return 8;
        case DType::Float64: return 8;
    }
}

// Lightweight GPU tensor — wraps an MTLBuffer with shape metadata.
class MTensor {
public:
    MTensor() = default;

#ifdef __OBJC__
    // GPU allocation (Objective-C++ only)
    MTensor(id<MTLDevice> device, std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        if (bytes == 0) bytes = 4;
        id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        _buffer = (void*)buf;
        _data = [buf contents];  // cache CPU-accessible pointer for C++ access
    }

    id<MTLBuffer> buffer() const { return (id<MTLBuffer>)_buffer; }
#endif

    // CPU allocation (no Metal buffer)
    MTensor(std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        _cpu_data.resize(bytes);
    }

    ~MTensor() {
        reset();
    }

    MTensor(const MTensor &other)
        : _buffer(other._buffer),
          _data(other._data),
          _cpu_data(other._cpu_data),
          _shape(other._shape),
          _dtype(other._dtype),
          _numel(other._numel) {
        retainBuffer(_buffer);
        if (!_buffer) _data = nullptr;
    }

    MTensor& operator=(const MTensor &other) {
        if (this == &other) return *this;
        reset();
        _buffer = other._buffer;
        _data = other._data;
        _cpu_data = other._cpu_data;
        _shape = other._shape;
        _dtype = other._dtype;
        _numel = other._numel;
        retainBuffer(_buffer);
        if (!_buffer) _data = nullptr;
        return *this;
    }

    MTensor(MTensor &&other) noexcept
        : _buffer(other._buffer),
          _data(other._data),
          _cpu_data(std::move(other._cpu_data)),
          _shape(std::move(other._shape)),
          _dtype(other._dtype),
          _numel(other._numel) {
        other._buffer = nullptr;
        other._data = nullptr;
        other._numel = 0;
    }

    MTensor& operator=(MTensor &&other) noexcept {
        if (this == &other) return *this;
        reset();
        _buffer = other._buffer;
        _data = other._data;
        _cpu_data = std::move(other._cpu_data);
        _shape = std::move(other._shape);
        _dtype = other._dtype;
        _numel = other._numel;
        other._buffer = nullptr;
        other._data = nullptr;
        other._numel = 0;
        return *this;
    }

    bool defined() const { return _buffer != nullptr || !_cpu_data.empty(); }
    bool isGpu() const { return _buffer != nullptr; }

    int64_t numel() const { return _numel; }
    int64_t size(int dim) const {
        if (dim < 0) dim += _shape.size();
        return _shape[dim];
    }
    int ndim() const { return (int)_shape.size(); }
    const std::vector<int64_t>& shape() const { return _shape; }
    DType dtype() const { return _dtype; }
    size_t elementSize() const { return dtypeSize(_dtype); }
    size_t nbytes() const { return _numel * dtypeSize(_dtype); }

    void* data_ptr() {
        if (_data) return _data;
        return _cpu_data.data();
    }
    const void* data_ptr() const {
        if (_data) return _data;
        return _cpu_data.data();
    }

    template<typename T> T* data() { return static_cast<T*>(data_ptr()); }
    template<typename T> const T* data() const { return static_cast<const T*>(data_ptr()); }

    void zero() {
        memset(data_ptr(), 0, _numel * dtypeSize(_dtype));
    }

    // Create a CPU copy of the data
    MTensor cpu() const {
        MTensor out(_shape, _dtype);
        memcpy(out.data_ptr(), data_ptr(), nbytes());
        return out;
    }

    void reset() {
        releaseBuffer(_buffer);
        _buffer = nullptr;
        _data = nullptr;
        _cpu_data.clear();
        _shape.clear();
        _numel = 0;
    }

    // Stride for dim 0 (elements per row)
    int64_t stride0() const {
        if (_shape.size() <= 1) return 1;
        int64_t s = 1;
        for (size_t i = 1; i < _shape.size(); i++) s *= _shape[i];
        return s;
    }

    // Create a view of the first `n` elements along dim 0.
    // GPU views retain the shared MTLBuffer, so the parent may be reset safely.
    MTensor view(int64_t n) const {
        assert(!_shape.empty());
        MTensor v;
        v._buffer = _buffer;  // shares the retained buffer
        v._data = _data;      // shares the CPU-accessible pointer
        v._cpu_data = _cpu_data;
        v._shape = _shape;
        v._shape[0] = n;
        v._dtype = _dtype;
        v._numel = n * stride0();
        retainBuffer(v._buffer);
        if (!v._buffer) v._data = nullptr;
        return v;
    }

private:
    static void retainBuffer(void* buffer) {
#if defined(__APPLE__)
        if (buffer) CFRetain(static_cast<CFTypeRef>(buffer));
#else
        (void)buffer;
#endif
    }

    static void releaseBuffer(void* buffer) {
#if defined(__APPLE__)
        if (buffer) CFRelease(static_cast<CFTypeRef>(buffer));
#else
        (void)buffer;
#endif
    }

    void* _buffer = nullptr;  // retained id<MTLBuffer> as void*
    void* _data = nullptr;    // cached CPU-accessible pointer (shared memory on Apple Silicon)
    std::vector<uint8_t> _cpu_data;
    std::vector<int64_t> _shape;
    DType _dtype = DType::Float32;
    int64_t _numel = 0;
};

// Factory helpers (Objective-C++ only)
#ifdef __OBJC__
inline MTensor mtensor_empty(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    return MTensor(dev, std::move(shape), dt);
}

inline MTensor mtensor_zeros(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    MTensor t(dev, std::move(shape), dt);
    t.zero();
    return t;
}
#endif

#endif
