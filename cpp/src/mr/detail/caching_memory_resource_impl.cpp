/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/mr/detail/caching_memory_resource_impl.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

caching_memory_resource_impl::caching_memory_resource_impl(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::optional<std::size_t> max_split_size)
  : upstream_mr_{std::move(upstream)}, max_split_size_{max_split_size}
{
}

caching_memory_resource_impl::~caching_memory_resource_impl() { (void)release(); }

device_async_resource_ref caching_memory_resource_impl::get_upstream_resource() const noexcept
{
  return device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
}

std::size_t caching_memory_resource_impl::release() noexcept
{
  lock_guard lock(this->get_mutex());
  return release_pool(true) + release_pool(false);
}

std::size_t caching_memory_resource_impl::release_small_blocks() noexcept
{
  lock_guard lock(this->get_mutex());
  return release_pool(true);
}

std::size_t caching_memory_resource_impl::release_large_blocks() noexcept
{
  lock_guard lock(this->get_mutex());
  return release_pool(false);
}

std::size_t caching_memory_resource_impl::cached_small_bytes() const noexcept
{
  return cached_small_bytes_;
}

std::size_t caching_memory_resource_impl::cached_large_bytes() const noexcept
{
  return cached_large_bytes_;
}

std::size_t caching_memory_resource_impl::upstream_allocation_size(std::size_t bytes) const noexcept
{
  if (bytes == 0) { return 0; }
  if (bytes <= small_size_threshold) { return small_segment_size; }
  if (bytes < minimum_large_allocation) { return large_segment_size; }
  return rmm::align_up(bytes, large_rounding);
}

std::size_t caching_memory_resource_impl::get_maximum_allocation_size() const
{
  return std::numeric_limits<std::size_t>::max();
}

caching_memory_resource_impl::block_type caching_memory_resource_impl::expand_pool(
  std::size_t size, free_list&, cuda_stream_view stream)
{
  auto const is_small      = is_small_allocation(size);
  auto const upstream_size = upstream_allocation_size(size);

  try {
    return block_from_upstream(upstream_size, stream, is_small);
  } catch (rmm::out_of_memory const&) {
    auto const released = release_pool(is_small);
    if (released == 0) { throw; }
    return block_from_upstream(upstream_size, stream, is_small);
  }
}

caching_memory_resource_impl::split_block caching_memory_resource_impl::allocate_from_block(
  block_type const& block, std::size_t size)
{
  releasable_blocks_.erase(block.pointer());

  auto const alloc = block_type{block.pointer(), size, block.is_head()};
  if (!should_split(block, size)) { return {block, {}}; }

  auto rest = block_type{block.pointer() + size, block.size() - size, false};
  return {alloc, rest};
}

caching_memory_resource_impl::block_type caching_memory_resource_impl::free_block(
  void* ptr, std::size_t size) noexcept
{
  auto const aligned_size = rmm::align_up(size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const block        = block_type{static_cast<char*>(ptr), aligned_size, false};

  auto const iter = upstream_blocks_.upper_bound(segment{static_cast<char*>(ptr), 0, false});
  if (iter != upstream_blocks_.begin()) {
    auto const candidate   = std::prev(iter);
    auto const segment_end = candidate->pointer + candidate->size;
    if (candidate->pointer <= static_cast<char*>(ptr) && static_cast<char*>(ptr) < segment_end) {
      return block_type{
        static_cast<char*>(ptr), aligned_size, candidate->pointer == static_cast<char*>(ptr)};
    }
  }

  return block;
}

std::pair<std::size_t, std::size_t> caching_memory_resource_impl::free_list_summary(
  free_list const& blocks)
{
  std::size_t largest{};
  std::size_t total{};
  std::for_each(blocks.cbegin(), blocks.cend(), [&largest, &total](auto const& block) {
    total += block.size();
    largest = std::max(largest, block.size());
  });
  return {largest, total};
}

bool caching_memory_resource_impl::is_small_allocation(std::size_t size) const noexcept
{
  return size <= small_size_threshold;
}

bool caching_memory_resource_impl::should_split(block_type const& block,
                                                std::size_t size) const noexcept
{
  auto const remaining = block.size() - size;
  if (block_is_small(block)) { return remaining >= minimum_block_size; }
  if (!max_split_size_.has_value()) { return remaining >= minimum_block_size; }
  if (block.size() >= max_split_size_.value() && size < max_split_size_.value()) { return false; }
  return remaining > small_size_threshold;
}

bool caching_memory_resource_impl::block_is_small(block_type const& block) const noexcept
{
  auto const iter = upstream_blocks_.upper_bound(segment{block.pointer(), 0, false});
  if (iter == upstream_blocks_.begin()) { return block.size() <= small_size_threshold; }

  auto const candidate   = std::prev(iter);
  auto const segment_end = candidate->pointer + candidate->size;
  if (candidate->pointer <= block.pointer() && block.pointer() < segment_end) {
    return candidate->is_small;
  }

  return block.size() <= small_size_threshold;
}

std::size_t caching_memory_resource_impl::release_pool(bool is_small) noexcept
{
  for (auto const& upstream : upstream_blocks_) {
    if (upstream.is_small != is_small) { continue; }
    releasable_blocks_[upstream.pointer] = released_block{upstream.size, upstream.is_small};
  }

  std::size_t released{};

  for (auto it = releasable_blocks_.begin(); it != releasable_blocks_.end();) {
    auto const [ptr, info] = *it;
    if (info.is_small != is_small) {
      ++it;
      continue;
    }

    auto const segment_it = upstream_blocks_.find(segment{ptr, 0, is_small});
    if (segment_it == upstream_blocks_.end() || segment_it->size != info.size ||
        segment_it->is_small != is_small) {
      it = releasable_blocks_.erase(it);
      continue;
    }

    get_upstream_resource().deallocate_sync(ptr, info.size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    released += info.size;
    if (is_small) {
      cached_small_bytes_ -= info.size;
    } else {
      cached_large_bytes_ -= info.size;
    }
    upstream_blocks_.erase(segment_it);
    it = releasable_blocks_.erase(it);
  }

  return released;
}

caching_memory_resource_impl::block_type caching_memory_resource_impl::block_from_upstream(
  std::size_t size, cuda_stream_view stream, bool is_small)
{
  void* ptr = get_upstream_resource().allocate(stream, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  upstream_blocks_.insert(segment{static_cast<char*>(ptr), size, is_small});
  if (is_small) {
    cached_small_bytes_ += size;
  } else {
    cached_large_bytes_ += size;
  }
  return block_type{static_cast<char*>(ptr), size, true};
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
