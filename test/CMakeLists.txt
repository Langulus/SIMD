add_executable(LangulusSIMDTest
	Main.cpp
	TestAdd.cpp
	TestDiv.cpp
	TestEqual.cpp
	TestMul.cpp
	TestPow.cpp
	TestShiftLeft.cpp
	TestShiftRight.cpp
	TestSub.cpp
)

target_link_libraries(LangulusSIMDTest
	PRIVATE		LangulusSIMD
				Catch2 
)

add_test(
	NAME		LangulusSIMDTest
	COMMAND		LangulusSIMDTest
)

langulus_copy_dlls(LangulusSIMDTest ON LangulusSIMDTest)
