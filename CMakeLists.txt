cmake_minimum_required(VERSION 3.28)
project(LangulusSIMD
    VERSION         1.0.0
    DESCRIPTION     "Langulus SIMD templates layer based on SIMDe library"
    HOMEPAGE_URL    https://langulus.com
)

# Check if this project is built as standalone, or a part of something else 
if(PROJECT_IS_TOP_LEVEL OR NOT LANGULUS)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

    include(LangulusUtilities.cmake)

    # Add Langulus::Core/Logger/RTTI libraries                              
    fetch_langulus_module(Core   GIT_TAG 393d95b6ab13fc5846bda8a3044e4ca62f02fd5a)
    fetch_langulus_module(Logger GIT_TAG 9c0156a1310b6734fea73a26f93aa532adfa1515)
    fetch_langulus_module(RTTI   GIT_TAG 699bbb5833690cffea20e791fee7a31669b929bc)
endif()

# Configure SIMDe library                                                   
fetch_external_module(
    SIMDe
    GIT_REPOSITORY  https://github.com/Epixu/simde.git
    GIT_TAG         no_windows_h
)

# Build and install SIMD library                                            
add_library(LangulusSIMD ${LANGULUS_LIBRARY_TYPE}
    $<TARGET_OBJECTS:LangulusLogger>
    $<TARGET_OBJECTS:LangulusRTTI>
)

if(MSVC)
	# Some SVML functions seem to be broken on x86 MSVC builds				
	# See https://github.com/simd-everywhere/simde/issues/1111				
    target_compile_options(LangulusSIMD INTERFACE -DSIMDE_X86_SVML_NO_NATIVE)
endif()

target_include_directories(LangulusSIMD
	PUBLIC      include
                $<TARGET_PROPERTY:LangulusLogger,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:LangulusRTTI,INTERFACE_INCLUDE_DIRECTORIES>
                ${SIMDe_SOURCE_DIR}
)

target_link_libraries(LangulusSIMD
    PUBLIC      LangulusCore
                fmt
)

target_compile_definitions(LangulusSIMD
    PRIVATE     LANGULUS_EXPORT_ALL
)

if(LANGULUS_TESTING)
    enable_testing()
	add_subdirectory(test)
endif()