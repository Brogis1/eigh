#ifndef EIGH_FFI_HELPERS_H_
#define EIGH_FFI_HELPERS_H_

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace eigh {

// ---------------------------------------------------------------------------
// FFI Buffer accessor compatibility shim.
//
// The XLA FFI typed-Buffer accessors were renamed across jaxlib versions:
//   * jaxlib >= 0.5 (FFI API minor >= 1): methods
//       buf.typed_data(), buf.untyped_data(), buf.dimensions(), buf.element_count()
//   * jaxlib  ~0.4.29 (FFI API minor 0): plain struct with public members
//       buf.data, buf.dimensions   (no typed_data()/element_count() methods)
//
// These helpers detect at compile time (via the detection idiom — no version
// macro needed) which form the Buffer exposes and use it. The compiled code is
// identical on modern jaxlib; only old jaxlib takes the member path. This lets
// the kernels build from source against either jaxlib generation unchanged.
// ---------------------------------------------------------------------------
namespace ffi_compat {

// Detect the jaxlib >= 0.5 buffer API by the presence of `untyped_data()`. This
// is a NON-template method on both the typed `Buffer<dtype>` and `AnyBuffer` in
// 0.5+, and absent on the 0.4.29 plain-struct buffers. We key the modern/old
// branch on this rather than on `typed_data()`, because on `AnyBuffer`
// `typed_data` is a TEMPLATE method — `decltype(buf.typed_data())` fails to
// compile (needs explicit <T>), which would wrongly select the old-API branch.
template <typename T, typename = void>
struct has_untyped_data : std::false_type {};
template <typename T>
struct has_untyped_data<
    T, std::void_t<decltype(std::declval<T&>().untyped_data())>>
    : std::true_type {};

// Detect a non-template `typed_data()` (the typed Buffer<dtype> in 0.5+).
template <typename T, typename = void>
struct has_typed_data_method : std::false_type {};
template <typename T>
struct has_typed_data_method<
    T, std::void_t<decltype(std::declval<T&>().typed_data())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_dimensions_method : std::false_type {};
template <typename T>
struct has_dimensions_method<
    T, std::void_t<decltype(std::declval<T&>().dimensions())>>
    : std::true_type {};

}  // namespace ffi_compat

// Typed data pointer. On 0.5+ the typed Buffer<dtype> exposes a non-template
// typed_data(); on 0.4.29 it is the `.data` member. (Only ever called on the
// typed Buffer, not AnyBuffer.)
template <typename Buf>
inline auto FfiTypedData(Buf& buf) {
    if constexpr (ffi_compat::has_typed_data_method<Buf>::value) {
        return buf.typed_data();
    } else {
        return buf.data;
    }
}

// Untyped (void*) data pointer: buf.untyped_data() (>=0.5) or (void*)buf.data.
template <typename Buf>
inline void* FfiUntypedData(Buf& buf) {
    if constexpr (ffi_compat::has_untyped_data<Buf>::value) {
        return buf.untyped_data();
    } else {
        return static_cast<void*>(buf.data);
    }
}

// Dimensions span: buf.dimensions() (>=0.5) or buf.dimensions member (0.4.29).
template <typename Buf>
inline ::xla::ffi::Span<const int64_t> FfiDimensions(Buf& buf) {
    if constexpr (ffi_compat::has_dimensions_method<Buf>::value) {
        return buf.dimensions();
    } else {
        return buf.dimensions;
    }
}

// Element count: buf.element_count() (>=0.5) or product of dimensions (0.4.29).
template <typename Buf>
inline int64_t FfiElementCount(Buf& buf) {
    auto dims = FfiDimensions(buf);
    return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                           std::multiplies<int64_t>());
}

namespace ffi_compat {

template <typename T, typename = void>
struct has_element_type_method : std::false_type {};
template <typename T>
struct has_element_type_method<
    T, std::void_t<decltype(std::declval<T&>().element_type())>>
    : std::true_type {};

}  // namespace ffi_compat

// Buffer element dtype: buf.element_type() (>=0.5) or buf.dtype member (0.4.29).
template <typename Buf>
inline ::xla::ffi::DataType FfiElementType(Buf& buf) {
    if constexpr (ffi_compat::has_element_type_method<Buf>::value) {
        return buf.element_type();
    } else {
        return buf.dtype;
    }
}

// InvalidArgument error: ffi::Error::InvalidArgument (>=0.5) vs the
// (ErrorCode, message) constructor (0.4.29, which lacks the static factory).
inline ::xla::ffi::Error FfiInvalidArgument(std::string message) {
    return ::xla::ffi::Error(::xla::ffi::ErrorCode::kInvalidArgument,
                             std::move(message));
}

// Byte width of an FFI DataType (covers the dtypes this package uses).
inline int64_t FfiByteWidth(::xla::ffi::DataType dtype) {
    using DT = ::xla::ffi::DataType;
    switch (dtype) {
        case DT::S32: return 4;
        case DT::F32: return 4;
        case DT::F64: return 8;
        case DT::C64: return 8;
        case DT::C128: return 16;
        default: return 0;
    }
}

namespace ffi_compat {

template <typename T, typename = void>
struct has_size_bytes_method : std::false_type {};
template <typename T>
struct has_size_bytes_method<
    T, std::void_t<decltype(std::declval<T&>().size_bytes())>>
    : std::true_type {};

}  // namespace ffi_compat

// Buffer size in bytes: buf.size_bytes() (>=0.5) or bytewidth * element_count
// (0.4.29, where AnyBuffer is a plain struct with no size_bytes()).
template <typename Buf>
inline int64_t FfiSizeBytes(Buf& buf) {
    if constexpr (ffi_compat::has_size_bytes_method<Buf>::value) {
        return buf.size_bytes();
    } else {
        return FfiByteWidth(FfiElementType(buf)) * FfiElementCount(buf);
    }
}

// Namespace-level DataType aliases (ffi::F32, ...) exist from jaxlib 0.5; on
// 0.4.29 only DataType::F32 exists. Provide eigh::dt::F32 etc. that work on both
// so kernels can reference dtypes without depending on the aliases.
namespace dt {
inline constexpr ::xla::ffi::DataType F32 = ::xla::ffi::DataType::F32;
inline constexpr ::xla::ffi::DataType F64 = ::xla::ffi::DataType::F64;
inline constexpr ::xla::ffi::DataType C64 = ::xla::ffi::DataType::C64;
inline constexpr ::xla::ffi::DataType C128 = ::xla::ffi::DataType::C128;
inline constexpr ::xla::ffi::DataType S32 = ::xla::ffi::DataType::S32;
}  // namespace dt

template <typename T>
inline T MaybeCastNoOverflow(std::int64_t value)
{
    if constexpr (sizeof(T) == sizeof(std::int64_t)) {
        return value;
    } else {
        if (value > std::numeric_limits<T>::max()) {
            throw std::runtime_error("overflow when casting " +
                                     std::to_string(value) + " to " +
                                     typeid(T).name());
        }
        return static_cast<T>(value);
    }
}

template <::xla::ffi::DataType dtype>
auto AllocateScratchMemory(std::size_t size)
    -> std::unique_ptr<std::remove_extent_t<::xla::ffi::NativeType<dtype>>[]>
{
    using ValueType = std::remove_extent_t<::xla::ffi::NativeType<dtype>>;
    return std::unique_ptr<ValueType[]>(new ValueType[size]);
}

template <typename T>
inline auto AllocateWorkspace(::xla::ffi::ScratchAllocator& scratch,
                              int64_t size, std::string_view name)
{
    auto maybe_workspace = scratch.Allocate(sizeof(T) * size);
    if (!maybe_workspace.has_value()) {
        throw std::runtime_error("Unable to allocate workspace for " +
                                 std::string(name));
    }
    return static_cast<T*>(maybe_workspace.value());
}

inline int64_t GetBatchSize(::xla::ffi::Span<const int64_t> dims) {
    return std::accumulate(dims.begin(), dims.end(),
                           1LL, std::multiplies<int64_t>());
}

inline std::tuple<int64_t, int64_t, int64_t> SplitBatch2D(
    ::xla::ffi::Span<const int64_t> dims)
{
    auto trailingDims = dims.last(2);
    return std::make_tuple(GetBatchSize(dims.first(dims.size() - 2)),
                         trailingDims.front(), trailingDims.back());
}

} // namespace eigh

#endif
