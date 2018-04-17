# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.1)

project(lda CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(SYSTEM
    ../third_party/json-3.1.2
    ../third_party/docopt.cpp-0.6.2
    ../third_party/xtensor-0.15.9/include
    ../third_party/xtl-0.4.7/include
    ../third_party/xsimd-4.1.2/include
)

add_definitions(
    -DXTENSOR_USE_XSIMD
    -DDOCOPT_HEADER_ONLY
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wpedantic \
        -Wconversion -Wsign-conversion -Wshadow -Wno-missing-braces")
endif()

add_executable(lda
    main.cc

    ../lda/lda.cc
    ../lda/lda_io.cc
    ../tsv/tsv.cc
)
