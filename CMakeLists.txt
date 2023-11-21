cmake_minimum_required(VERSION 3.24)
project(LangulusSIMD
    VERSION         1.0.0
    DESCRIPTION     "Langulus SIMD templates layer based on SIMDe library"
    HOMEPAGE_URL    https://langulus.com
)

# Check if this project is built as standalone, or a part of something else 
if(PROJECT_IS_TOP_LEVEL OR NOT LANGULUS)
    include(LangulusUtilities.cmake)

    # Add Langulus::Core/Logger/RTTI libraries                              
    fetch_langulus_module(Core   GIT_TAG 0097bd180a5c625f88db0db0514bbf93aff53c7a)
    fetch_langulus_module(Logger GIT_TAG b63513abb9fb3014107335949e17e41e6e8a334b)
    fetch_langulus_module(RTTI   GIT_TAG f7c610255b819b99a6adcdefa970387490093c25)
endif()

# Configure SIMDe library                                                   
fetch_external_module(
    SIMDe
    GIT_REPOSITORY  https://github.com/simd-everywhere/simde.git
    GIT_TAG         cb9c8081e5ddcd59e3cb010ddaf66736ea99f1b7 # master branch
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