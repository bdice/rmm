/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>

#include <gtest/gtest.h>

class raii_restore_env {
 public:
  raii_restore_env(char const* name) : name_(name)
  {
    auto const value_or_null = getenv(name);
    if (value_or_null != nullptr) {
      value_  = value_or_null;
      is_set_ = true;
    }
  }

  ~raii_restore_env()
  {
    if (is_set_) {
      setenv(name_.c_str(), value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_{};
  std::string value_{};
  bool is_set_{false};
};

TEST(Adaptor, first)
{
  rmm::mr::cuda_memory_resource upstream;

  rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource> log_mr{&upstream,
                                                                          "logs/test1.txt"};

  auto p = log_mr.allocate(100);
  log_mr.deallocate(p, 100);
}

TEST(Adaptor, factory)
{
  rmm::mr::cuda_memory_resource upstream;

  auto log_mr = rmm::mr::make_logging_adaptor(&upstream, "logs/test2.txt");

  auto p = log_mr.allocate(100);
  log_mr.deallocate(p, 100);
}

TEST(Adaptor, EnviromentPath)
{
  rmm::mr::cuda_memory_resource upstream;

  // restore the original value (or unset) after test
  raii_restore_env old_env("RMM_LOG_FILE");

  unsetenv("RMM_LOG_FILE");

  // expect logging adaptor to fail if RMM_LOG_FILE is unset
  EXPECT_THROW(rmm::mr::make_logging_adaptor(&upstream), rmm::logic_error);

  setenv("RMM_LOG_FILE", "envtest.txt", 1);

  // use log file location specified in environment variable RMM_LOG_FILE
  auto log_mr = rmm::mr::make_logging_adaptor(&upstream);

  auto p = log_mr.allocate(100);
  log_mr.deallocate(p, 100);
}

TEST(Adaptor, STDOUT)
{
  testing::internal::CaptureStdout();

  rmm::mr::cuda_memory_resource upstream;

  auto log_mr = rmm::mr::make_logging_adaptor(&upstream, std::cout);

  auto p = log_mr.allocate(100);
  log_mr.deallocate(p, 100);

  std::string output = testing::internal::GetCapturedStdout();
  std::string header = output.substr(0, output.find("\n"));
  ASSERT_EQ(header, "Time,Action,Pointer,Size,Stream");
}

TEST(Adaptor, STDERR)
{
  testing::internal::CaptureStderr();

  rmm::mr::cuda_memory_resource upstream;

  auto log_mr = rmm::mr::make_logging_adaptor(&upstream, std::cerr);

  auto p = log_mr.allocate(100);
  log_mr.deallocate(p, 100);

  std::string output = testing::internal::GetCapturedStderr();
  std::string header = output.substr(0, output.find("\n"));
  ASSERT_EQ(header, "Time,Action,Pointer,Size,Stream");
}
