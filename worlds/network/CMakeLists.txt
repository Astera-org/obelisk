cmake_minimum_required(VERSION 3.23)

project(agent)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)

# thrift
FetchContent_Declare(
  thrift
  URL https://github.com/apache/thrift/archive/refs/tags/v0.16.0.zip
)
FetchContent_MakeAvailable(thrift)
list(APPEND CMAKE_MODULE_PATH "${thrift_SOURCE_DIR}/build/cmake/")
include(ThriftMacros)

include_directories("${thrift_SOURCE_DIR}/lib/cpp/src")
include_directories("${thrift_BINARY_DIR}")

set(SOURCES
  gencpp/Agent.cpp
  gencpp/env_types.cpp
  cpp/thrift_agent_client.cpp
  )

add_library(agent STATIC ${SOURCES})
set_target_properties(agent PROPERTIES PUBLIC_HEADER cpp/thrift_agent_client.h)
target_link_libraries(agent thrift)
target_include_directories(agent PUBLIC gencpp/)

# uncomment these to create a sample program
#add_executable(thrift_agent_client cpp/thrift_agent_client.cpp)
#target_include_directories(thrift_agent_client PRIVATE gencpp/)

#target_link_libraries(thrift_agent_client agent)
