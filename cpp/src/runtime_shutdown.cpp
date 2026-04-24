/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/runtime_shutdown.hpp>
#include <rmm/process_is_exiting.hpp>

#include <atomic>
#include <cstdlib>
#include <mutex>

namespace RMM_NAMESPACE {

namespace {

std::atomic<bool> exiting{false};
std::once_flag registered;

}  // namespace

bool process_is_exiting() noexcept { return exiting.load(std::memory_order_acquire); }

namespace detail {

void register_process_exit_hook() noexcept
{
  std::call_once(registered, []() {
    std::atexit([]() noexcept { exiting.store(true, std::memory_order_release); });
  });
}

}  // namespace detail
}  // namespace RMM_NAMESPACE
