/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/error.hpp>
#include <rmm/mr/caching_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using caching_mr  = rmm::mr::caching_memory_resource;
using cuda_mr     = rmm::mr::cuda_memory_resource;
using limiting_mr = rmm::mr::limiting_resource_adaptor;

TEST(CachingMemoryResourceTest, UpstreamSizingPolicy)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref()};
  EXPECT_EQ(mr.upstream_allocation_size(1), 2UL << 20);
  EXPECT_EQ(mr.upstream_allocation_size(1UL << 20), 2UL << 20);
  EXPECT_EQ(mr.upstream_allocation_size((1UL << 20) + 1), 20UL << 20);
  EXPECT_EQ(mr.upstream_allocation_size((10UL << 20) + 1), 12UL << 20);
}

TEST(CachingMemoryResourceTest, ReusesSmallAllocation)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref()};
  auto* ptr1 = mr.allocate_sync(4096);
  ASSERT_NE(ptr1, nullptr);
  mr.deallocate_sync(ptr1, 4096);
  auto* ptr2 = mr.allocate_sync(4096);
  EXPECT_EQ(ptr1, ptr2);
  mr.deallocate_sync(ptr2, 4096);
}

TEST(CachingMemoryResourceTest, ReleasesCachedBlocksOnFailure)
{
  cuda_mr cuda;
  limiting_mr limiter{cuda, 22UL << 20};
  caching_mr mr{limiter};

  auto* small = mr.allocate_sync(4096);
  mr.deallocate_sync(small, 4096);

  EXPECT_EQ(mr.cached_small_bytes(), 2UL << 20);
  EXPECT_NO_THROW((void)mr.allocate_sync(20UL << 20));
}

TEST(CachingMemoryResourceTest, ReleaseDropsCachedBytes)
{
  caching_mr mr{rmm::mr::get_current_device_resource_ref()};
  auto* ptr = mr.allocate_sync(4096);
  mr.deallocate_sync(ptr, 4096);
  EXPECT_GT(mr.cached_small_bytes(), 0);
  EXPECT_GT(mr.release_small_blocks(), 0);
  EXPECT_EQ(mr.cached_small_bytes(), 0);
}

}  // namespace

namespace test_properties {

static_assert(
  cuda::mr::resource_with<rmm::mr::caching_memory_resource, cuda::mr::device_accessible>);

}  // namespace test_properties

}  // namespace rmm::test
