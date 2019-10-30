cmake_minimum_required(VERSION 2.8.12)

project(oak2)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(BasicConfig)

cond_find_package(mia REQUIRED)

find_package(TBB REQUIRED)
find_package(Boost COMPONENTS serialization REQUIRED)

file(GLOB_RECURSE OAK_SRC "src/*.cpp")
file(GLOB_RECURSE OAK_HDR "include/*.h")
file(GLOB_RECURSE OAK_HPP "include/*.hpp")
list(APPEND OAK_SRC ${OAK_HDR})
list(APPEND OAK_SRC ${OAK_HPP})

add_library(${PROJECT_NAME} STATIC ${OAK_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_PDB_NAME_DEBUG "${PROJECT_NAME}${CMAKE_DEBUG_POSTFIX}")

target_include_directories(${PROJECT_NAME} PUBLIC "include")
target_include_directories(${PROJECT_NAME} PUBLIC ${TBB_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC "mia")
target_link_libraries(${PROJECT_NAME} PUBLIC ${TBB_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_SERIALIZATION_LIBRARY})

export(PACKAGE oak2)
export(TARGETS oak2
  FILE "${PROJECT_BINARY_DIR}/oak2Config.cmake"
)
