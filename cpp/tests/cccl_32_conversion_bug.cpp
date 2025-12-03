/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal reproducer for CCCL 3.2.x resource_ref conversion bug
// This reproduces the exact pattern from cudf's device_scalar that causes segfault

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/host_vector.h>

#include <gtest/gtest.h>

// Simplified rmm_host_allocator (like cudf's)
template <typename T>
class simple_host_allocator {
 public:
  using value_type = T;

  simple_host_allocator() = delete;

  // Constructor takes ANY host_async_resource_ref (with any properties)
  template <class... Properties>
  simple_host_allocator(cuda::mr::resource_ref<cuda::mr::host_accessible, Properties...> mr,
                        rmm::cuda_stream_view stream)
    : mr_(mr),  // Implicit conversion happens here!
      stream_(stream)
  {
  }

  simple_host_allocator(simple_host_allocator const&) = default;
  simple_host_allocator(simple_host_allocator&&)      = default;

  T* allocate(std::size_t n)
  {
    auto const result = mr_.allocate(stream_, n * sizeof(T));
    stream_.synchronize();
    return static_cast<T*>(result);
  }

  void deallocate(T* p, std::size_t n) noexcept
  {
    // This is where the segfault happens - mr_ has corrupt internal state
    mr_.deallocate(stream_, p, n * sizeof(T));
  }

  bool operator==(simple_host_allocator const& other) const { return mr_ == other.mr_; }
  bool operator!=(simple_host_allocator const& other) const { return !(*this == other); }

 private:
  rmm::host_async_resource_ref mr_;  // NOTE: Only host_accessible, not device_accessible!
  rmm::cuda_stream_view stream_;
};

// Function that returns host_device_async_resource_ref (like cudf::get_pinned_memory_resource)
rmm::host_device_async_resource_ref get_test_resource()
{
  static rmm::mr::pinned_host_memory_resource mr{};
  return rmm::host_device_async_resource_ref{mr};  // Returns with both host+device properties
}

// Mimics cudf's make_pinned_vector pattern
template <typename T>
thrust::host_vector<T, simple_host_allocator<T>> make_test_vector(std::size_t size,
                                                                  rmm::cuda_stream_view stream)
{
  // This triggers the conversion: host_device_async_resource_ref → host_async_resource_ref
  return thrust::host_vector<T, simple_host_allocator<T>>(
    size, simple_host_allocator<T>{get_test_resource(), stream});
}

// Test that reproduces the exact cudf pattern
TEST(CCCL32ConversionBug, DeviceScalarBounceBufferPattern)
{
  rmm::cuda_stream stream{};

  // Simulate device_scalar construction:
  // bounce_buffer{make_pinned_vector<T>(1, stream)}
  auto bounce_buffer = make_test_vector<int>(1, stream);

  // Use it
  bounce_buffer[0] = 42;
  ASSERT_EQ(bounce_buffer[0], 42);

  // Destructor will be called here - this should segfault with the bug
}

// Simpler direct conversion test
TEST(CCCL32ConversionBug, DirectConversion)
{
  rmm::cuda_stream stream{};

  // Get a resource with host+device properties
  auto mr_hostdevice = get_test_resource();

  // Create allocator (triggers conversion to host-only)
  simple_host_allocator<int> alloc{mr_hostdevice, stream};

  // Allocate
  int* ptr = alloc.allocate(10);
  ASSERT_NE(ptr, nullptr);

  // Deallocate - this should segfault with the bug
  alloc.deallocate(ptr, 10);
}
