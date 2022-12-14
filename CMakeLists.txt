cmake_minimum_required(VERSION 3.22)
project(Langulus.SIMD)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if this project is built as standalone, or a part of something else 
if(PROJECT_IS_TOP_LEVEL)
    include(LangulusUtilities.cmake)
    langulus_init_git_submodule(cmake)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(FetchLangulusModule)
    include(HandleLangulusFeatures)

    handle_langulus_features()

    fetch_langulus_module(Core)
    fetch_langulus_module(Logger)
    fetch_langulus_module(RTTI)

    add_compile_definitions(LANGULUS_ENABLE_TESTING)
    enable_testing()
	add_subdirectory(test)
endif()

# Configure SIMDe library                                                   
download_project(
	PROJ                SIMDe
	GIT_REPOSITORY      https://github.com/simd-everywhere/simde.git
	GIT_TAG             master
	UPDATE_DISCONNECTED 1
)

# Build and install SIMD library                                            
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/source LANGULUS_SIMD_SOURCES)
add_library(Langulus.SIMD INTERFACE)

target_include_directories(Langulus.SIMD
	INTERFACE   include
                ${SIMDe_SOURCE_DIR}
)

target_link_libraries(Langulus.SIMD
    INTERFACE   Langulus.Core
                Langulus.Logger
                Langulus.RTTI
)
