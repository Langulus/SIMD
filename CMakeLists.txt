cmake_minimum_required(VERSION 3.30)
project(LangulusSIMD
   VERSION         1.0.0
   DESCRIPTION     "Langulus SIMD templates layer based on SIMDe library"
   HOMEPAGE_URL    https://langulus.com
)

# Check if this project is built as standalone, or a part of something else 
if (PROJECT_IS_TOP_LEVEL OR NOT LANGULUS)
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
   set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
   set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
   include(LangulusUtilities.cmake)
   fetch_langulus_module(Logger GIT_TAG wip)
endif()

# Configure SIMDe library                                                   
fetch_external_module(
   SIMDe
   GIT_REPOSITORY  https://github.com/simd-everywhere/simde.git
   GIT_TAG         v0.8.4-rc3
)

# Build and install SIMD library                                            
add_langulus_library(LangulusSIMD 
   SOURCES  $<$<BOOL:${LANGULUS_OPTION_LOGGING}>:$<TARGET_OBJECTS:LangulusLogger>>
)

#if(MSVC)
   # Some SVML functions seem to be broken on x86 MSVC builds				
   # See https://github.com/simd-everywhere/simde/issues/1111				
#   target_compile_options(LangulusSIMD INTERFACE -DSIMDE_X86_SVML_NO_NATIVE)
#endif()

target_include_directories(LangulusSIMD
   PUBLIC   $<$<BOOL:${LANGULUS_OPTION_LOGGING}>:$<TARGET_PROPERTY:LangulusLogger,INTERFACE_INCLUDE_DIRECTORIES>>
            ${SIMDe_SOURCE_DIR}
            include
)

target_link_libraries(LangulusSIMD
   PUBLIC   LangulusCore
            LangulusCT
            $<$<BOOL:${LANGULUS_OPTION_LOGGING}>:fmt>
)

target_compile_definitions(LangulusSIMD
   PRIVATE  LANGULUS_EXPORT_ALL
)

if(LANGULUS_OPTION_TESTING)
   enable_testing()
   add_subdirectory(test)
endif()