cmake_minimum_required(VERSION 3.24)
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
    fetch_langulus_module(Core   GIT_TAG f1e134132b9dc3fbc8604d428a4d702c1405c256)
    fetch_langulus_module(Logger GIT_TAG 758350aae22de77eac65152d2e4b3ca74fb5d5ad)
    fetch_langulus_module(RTTI   GIT_TAG 3ff4a909d62be9891534fb4943d793fe62973ffc)
endif()

# Configure SIMDe library                                                   
fetch_external_module(
    SIMDe
    GIT_REPOSITORY  https://github.com/simd-everywhere/simde.git
    GIT_TAG         5e7c4d4ec10855ebe8f19af8a80fdfa26caad5e9 # master branch
)

# Build and install SIMD library                                            
add_library(LangulusSIMD ${LANGULUS_LIBRARY_TYPE}
    $<TARGET_OBJECTS:LangulusLogger>
    $<TARGET_OBJECTS:LangulusRTTI>
)

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