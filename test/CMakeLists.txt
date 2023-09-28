add_executable(LangulusSIMDTest
	Main.cpp
	TestAdd.cpp
	TestDiv.cpp
	TestEqual.cpp
	TestEqual2.cpp
	TestMul.cpp
	TestMul2.cpp
	TestPow.cpp
	TestShiftLeft.cpp
	TestShiftRight.cpp
	TestStrangeMSVCBug.cpp
	TestSub.cpp
	TestTypeCorrectness.cpp
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
