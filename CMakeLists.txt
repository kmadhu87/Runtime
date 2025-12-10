cmake_minimum_required(VERSION 3.15)
project(RuntimeProject CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_FLAGS -O3)

include_directories(include)

find_package(LLVM REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GPERFTOOLS REQUIRED libtcmalloc)

add_executable(runtime
    main.cpp
)

target_link_libraries(runtime 
	PRIVATE 
	LLVMSupport
	tcmalloc_minimal
	)

