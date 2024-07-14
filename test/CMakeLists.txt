file(GLOB_RECURSE
	LANGULUS_SIMD_TEST_SOURCES 
	LIST_DIRECTORIES FALSE CONFIGURE_DEPENDS
	*.cpp
)

add_executable(LangulusSIMDTest ${LANGULUS_SIMD_TEST_SOURCES})

if(MSVC)
    target_compile_options(LangulusCore INTERFACE -DSIMDE_X86_SVML_NO_NATIVE)
endif()

target_link_libraries(LangulusSIMDTest
	PRIVATE		LangulusSIMD
				Catch2 
)

add_test(
	NAME		LangulusSIMDTest
	COMMAND		LangulusSIMDTest
)