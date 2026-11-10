add_langulus_test(LangulusSIMDTest
   SOURCES     TestCeil.cpp
               TestDiv.cpp
               TestFloor.cpp
               TestMax.cpp
               TestMin.cpp
               TestPack.cpp
               TestPow.cpp
               TestShiftLeft.cpp
               TestShiftRight.cpp
               TestStrangeMSVCBug.cpp
               TestTypeCorrectness.cpp
               TestUnpack.cpp
               Add/TestAdd-VS.cpp
               Add/TestAdd-VV.cpp
               Equal/TestEqual-VVB.cpp
               Equal/TestEqual-VVM.cpp
               Multiply/TestMul-VS.cpp
               Multiply/TestMul-VV.cpp
               Subtract/TestSub-VS.cpp
               Subtract/TestSub-VV.cpp
   LIBRARIES   LangulusSIMD
)