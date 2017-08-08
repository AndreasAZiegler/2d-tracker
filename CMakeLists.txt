cmake_minimum_required(VERSION 2.8)

SET(CMAKE_BUILD_TYPE Release)
project(2d-tracker)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()

find_package(Eigen3 REQUIRED)

include_directories(include)

add_library(2d-tracker SHARED src/2d-tracker.cpp include/2d-tracker.h)
target_link_libraries (2d-tracker Eigen3::Eigen)

# Install library
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib/${PROJECT_NAME})

# Install library headers
file(GLOB HEADERS include/*.h)
install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME})
