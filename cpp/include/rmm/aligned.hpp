/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <rmm/detail/export.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace RMM_EXPORT rmm {

/**
 * @addtogroup utilities
 * @{
 * @file
 */

/**
 * @brief Default alignment used for host memory allocated by RMM.
 *
 */
static constexpr std::size_t RMM_DEFAULT_HOST_ALIGNMENT{alignof(std::max_align_t)};

/**
 * @brief Default alignment used for CUDA memory allocation.
 *
 */
static constexpr std::size_t CUDA_ALLOCATION_ALIGNMENT{256};

/**
 * @brief Returns whether or not `value` is a power of 2.
 *
 * @param[in] value value to check.
 *
 * @return True if the input is a power of two with non-negative integer exponent, false otherwise.
 */
[[nodiscard]] bool is_pow2(std::size_t value) noexcept;

/**
 * @brief Returns whether or not `alignment` is a valid memory alignment.
 *
 * @param[in] alignment to check
 *
 * @return True if the alignment is valid, false otherwise.
 */
[[nodiscard]] bool is_supported_alignment(std::size_t alignment) noexcept;

/**
 * @brief Align up to nearest multiple of specified power of 2
 *
 * @param[in] value value to align
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return the aligned value
 */
[[nodiscard]] std::size_t align_up(std::size_t value, std::size_t alignment) noexcept;

/**
 * @brief Align down to the nearest multiple of specified power of 2
 *
 * @param[in] value value to align
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return the aligned value
 */
[[nodiscard]] std::size_t align_down(std::size_t value, std::size_t alignment) noexcept;

/**
 * @brief Checks whether a value is aligned to a multiple of a specified power of 2
 *
 * @param[in] value value to check for alignment
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return true if aligned
 */
[[nodiscard]] bool is_aligned(std::size_t value, std::size_t alignment) noexcept;

/**
 * @brief Checks whether the provided pointer is aligned to a specified @p alignment
 *
 * @param[in] ptr pointer to check for alignment
 * @param[in] alignment required alignment in bytes, must be a power of 2
 *
 * @return true if the pointer is aligned
 */
[[nodiscard]] bool is_pointer_aligned(void* ptr,
                                      std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT) noexcept;

/** @} */  // end of group

}  // namespace RMM_EXPORT rmm
