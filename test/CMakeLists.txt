file(GLOB_RECURSE
	LANGULUS_SIMD_TEST_SOURCES 
	LIST_DIRECTORIES FALSE CONFIGURE_DEPENDS
	*.cpp
)

add_executable(LangulusSIMDTest ${LANGULUS_SIMD_TEST_SOURCES})

target_link_libraries(LangulusSIMDTest
	PRIVATE		LangulusSIMD
				Catch2 
)

add_test(
	NAME		LangulusSIMDTest
	COMMAND		LangulusSIMDTest
)