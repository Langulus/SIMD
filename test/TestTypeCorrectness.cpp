///                                                                           
/// Langulus::Math                                                            
/// Copyright (c) 2014 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "Common.hpp"


using Vec2u8  = Vector<::std::uint8_t,  2>;
using Vec2u16 = Vector<::std::uint16_t, 2>;
using Vec2u32 = Vector<::std::uint32_t, 2>;
using Vec2u64 = Vector<::std::uint64_t, 2>;

using Vec2i8  = Vector<::std::int8_t,   2>;
using Vec2i16 = Vector<::std::int16_t,  2>;
using Vec2i32 = Vector<::std::int32_t,  2>;
using Vec2i64 = Vector<::std::int64_t,  2>;

using Vec2f = Vector<float, 2>;
using Vec2d = Vector<double, 2>;


TEST_CASE_TEMPLATE("Testing vector checks", TestType, 
   Vec2u8, Vec2u16, Vec2u32, Vec2u64, 
   Vec2i8, Vec2i16, Vec2i32, Vec2i64, 
   Vec2f, Vec2d
) {
   static_assert(    CT::Vector<TestType>);
   static_assert(not CT::Scalar<TestType>);
   static_assert(ExtentOf<TestType> == 2);
}

TEST_CASE_TEMPLATE("Testing scalar checks", TestType, NUMBERS_ALL()) {
   static_assert(not CT::Vector<TestType>);
   static_assert(    CT::Scalar<TestType>);
   static_assert(ExtentOf<TestType> == 1);
}

TEST_CASE_TEMPLATE("Padding and alignment checks", T, NUMBERS_ALL()) {
   static_assert(sizeof(Vector<T, 1>) == sizeof(T) * 1);
   static_assert(sizeof(Vector<T, 2>) == sizeof(T) * 2);
   static_assert(sizeof(Vector<T, 3>) == sizeof(T) * 3);
   static_assert(sizeof(Vector<T, 4>) == sizeof(T) * 4);

   static_assert(sizeof(Vector<T, 1>[12]) == sizeof(Vector<T, 4>[3]));
   static_assert(sizeof(Vector<T, 1>[12]) == sizeof(Vector<T, 3>[4]));
   static_assert(sizeof(Vector<T, 1>[12]) == sizeof(Vector<T, 2>[6]));
   static_assert(sizeof(Vector<T, 2>[12]) == sizeof(Vector<T, 3>[8]));
   static_assert(sizeof(Vector<T, 2>[12]) == sizeof(Vector<T, 4>[6]));
   static_assert(sizeof(Vector<T, 3>[ 8]) == sizeof(Vector<T, 4>[6]));
}

TEST_CASE_TEMPLATE("ExtentOf checks", T, NUMBERS_ALL()) {
   static_assert(ExtentOf<T> == 1);

   static_assert(ExtentOf<Vector<T, 1>> == 1);
   static_assert(ExtentOf<Vector<T, 2>> == 2);
   static_assert(ExtentOf<Vector<T, 3>> == 3);
   static_assert(ExtentOf<Vector<T, 4>> == 4);

   static_assert(ExtentOf<Vector<T, 1>[12]> == 12);
   static_assert(ExtentOf<Vector<T, 1>[12]> == 12);
   static_assert(ExtentOf<Vector<T, 1>[12]> == 12);
   static_assert(ExtentOf<Vector<T, 2>[12]> == 12);
   static_assert(ExtentOf<Vector<T, 2>[12]> == 12);
   static_assert(ExtentOf<Vector<T, 3>[ 8]> ==  8);

   static_assert(AllExtentsOf<Vector<T, 2>[12]> == 24);
   static_assert(AllExtentsOf<Vector<T, 2>[12]> == 24);
   static_assert(AllExtentsOf<Vector<T, 3>[ 8]> == 24);
}

TEST_CASE("LglsOverlapExtents checks") {
   int scalar {};
   int scalarArray[1] {};
   int smallArray[2] {};
   int bigArray[4] {};

   Vector<int, 1> v1;
   Vector<int, 2> v2;
   Vector<int, 4> v4;

   Vector<int, 1> v1x2[2];
   Vector<int, 2> v2x2[2];
   Vector<int, 4> v4x2[2];

   static_assert(LglsOverlapExtents(scalar,      scalar     ) == 1);
   static_assert(LglsOverlapExtents(scalar,      scalarArray) == 1);
   static_assert(LglsOverlapExtents(scalar,      smallArray ) == 2);
   static_assert(LglsOverlapExtents(scalar,      bigArray   ) == 4);
   static_assert(LglsOverlapExtents(scalar,      v1         ) == 1);
   static_assert(LglsOverlapExtents(scalar,      v2         ) == 2);
   static_assert(LglsOverlapExtents(scalar,      v4         ) == 4);
   static_assert(LglsOverlapExtents(scalar,      v1x2       ) == 1*2);
   static_assert(LglsOverlapExtents(scalar,      v2x2       ) == 2*2);
   static_assert(LglsOverlapExtents(scalar,      v4x2       ) == 4*2);

   static_assert(LglsOverlapExtents(scalarArray, scalar     ) == 1);
   static_assert(LglsOverlapExtents(scalarArray, scalarArray) == 1);
   static_assert(LglsOverlapExtents(scalarArray, smallArray ) == 2);
   static_assert(LglsOverlapExtents(scalarArray, bigArray   ) == 4);
   static_assert(LglsOverlapExtents(scalarArray, v1         ) == 1);
   static_assert(LglsOverlapExtents(scalarArray, v2         ) == 2);
   static_assert(LglsOverlapExtents(scalarArray, v4         ) == 4);
   static_assert(LglsOverlapExtents(scalarArray, v1x2       ) == 1*2);
   static_assert(LglsOverlapExtents(scalarArray, v2x2       ) == 2*2);
   static_assert(LglsOverlapExtents(scalarArray, v4x2       ) == 4*2);

   static_assert(LglsOverlapExtents(smallArray,  scalar     ) == 2);
   static_assert(LglsOverlapExtents(smallArray,  scalarArray) == 2);
   static_assert(LglsOverlapExtents(smallArray,  smallArray ) == 2);
   static_assert(LglsOverlapExtents(smallArray,  bigArray   ) == 2);
   static_assert(LglsOverlapExtents(smallArray,  v1         ) == 2);
   static_assert(LglsOverlapExtents(smallArray,  v2         ) == 2);
   static_assert(LglsOverlapExtents(smallArray,  v4         ) == 2);
   static_assert(LglsOverlapExtents(smallArray,  v1x2       ) == 2);
   static_assert(LglsOverlapExtents(smallArray,  v2x2       ) == 2);
   static_assert(LglsOverlapExtents(smallArray,  v4x2       ) == 2);

   static_assert(LglsOverlapExtents(bigArray,    scalar     ) == 4);
   static_assert(LglsOverlapExtents(bigArray,    scalarArray) == 4);
   static_assert(LglsOverlapExtents(bigArray,    smallArray ) == 2);
   static_assert(LglsOverlapExtents(bigArray,    bigArray   ) == 4);
   static_assert(LglsOverlapExtents(bigArray,    v1         ) == 4);
   static_assert(LglsOverlapExtents(bigArray,    v2         ) == 2);
   static_assert(LglsOverlapExtents(bigArray,    v4         ) == 4);
   static_assert(LglsOverlapExtents(bigArray,    v1x2       ) == 2);
   static_assert(LglsOverlapExtents(bigArray,    v2x2       ) == 4);
   static_assert(LglsOverlapExtents(bigArray,    v4x2       ) == 4);

   static_assert(LglsOverlapExtents(v1,          scalar     ) == 1);
   static_assert(LglsOverlapExtents(v1,          scalarArray) == 1);
   static_assert(LglsOverlapExtents(v1,          smallArray ) == 2);
   static_assert(LglsOverlapExtents(v1,          bigArray   ) == 4);
   static_assert(LglsOverlapExtents(v1,          v1         ) == 1);
   static_assert(LglsOverlapExtents(v1,          v2         ) == 2);
   static_assert(LglsOverlapExtents(v1,          v4         ) == 4);
   static_assert(LglsOverlapExtents(v1,          v1x2       ) == 1*2);
   static_assert(LglsOverlapExtents(v1,          v2x2       ) == 2*2);
   static_assert(LglsOverlapExtents(v1,          v4x2       ) == 4*2);

   static_assert(LglsOverlapExtents(v2,          scalar     ) == 2);
   static_assert(LglsOverlapExtents(v2,          scalarArray) == 2);
   static_assert(LglsOverlapExtents(v2,          smallArray ) == 2);
   static_assert(LglsOverlapExtents(v2,          bigArray   ) == 2);
   static_assert(LglsOverlapExtents(v2,          v1         ) == 2);
   static_assert(LglsOverlapExtents(v2,          v2         ) == 2);
   static_assert(LglsOverlapExtents(v2,          v4         ) == 2);
   static_assert(LglsOverlapExtents(v2,          v1x2       ) == 2);
   static_assert(LglsOverlapExtents(v2,          v2x2       ) == 2);
   static_assert(LglsOverlapExtents(v2,          v4x2       ) == 2);

   static_assert(LglsOverlapExtents(v4,          scalar     ) == 4);
   static_assert(LglsOverlapExtents(v4,          scalarArray) == 4);
   static_assert(LglsOverlapExtents(v4,          smallArray ) == 2);
   static_assert(LglsOverlapExtents(v4,          bigArray   ) == 4);
   static_assert(LglsOverlapExtents(v4,          v1         ) == 4);
   static_assert(LglsOverlapExtents(v4,          v2         ) == 2);
   static_assert(LglsOverlapExtents(v4,          v4         ) == 4);
   static_assert(LglsOverlapExtents(v4,          v1x2       ) == 2);
   static_assert(LglsOverlapExtents(v4,          v2x2       ) == 4);
   static_assert(LglsOverlapExtents(v4,          v4x2       ) == 4);

   static_assert(LglsOverlapExtents(v1x2,        scalar     ) == 2);
   static_assert(LglsOverlapExtents(v1x2,        scalarArray) == 2);
   static_assert(LglsOverlapExtents(v1x2,        smallArray ) == 2);
   static_assert(LglsOverlapExtents(v1x2,        bigArray   ) == 2);
   static_assert(LglsOverlapExtents(v1x2,        v1         ) == 2);
   static_assert(LglsOverlapExtents(v1x2,        v2         ) == 2);
   static_assert(LglsOverlapExtents(v1x2,        v4         ) == 2);
   static_assert(LglsOverlapExtents(v1x2,        v1x2       ) == 2);
   static_assert(LglsOverlapExtents(v1x2,        v2x2       ) == 2);
   static_assert(LglsOverlapExtents(v1x2,        v4x2       ) == 2);

   static_assert(LglsOverlapExtents(v2x2,        scalar     ) == 4);
   static_assert(LglsOverlapExtents(v2x2,        scalarArray) == 4);
   static_assert(LglsOverlapExtents(v2x2,        smallArray ) == 2);
   static_assert(LglsOverlapExtents(v2x2,        bigArray   ) == 4);
   static_assert(LglsOverlapExtents(v2x2,        v1         ) == 4);
   static_assert(LglsOverlapExtents(v2x2,        v2         ) == 2);
   static_assert(LglsOverlapExtents(v2x2,        v4         ) == 4);
   static_assert(LglsOverlapExtents(v2x2,        v1x2       ) == 2);
   static_assert(LglsOverlapExtents(v2x2,        v2x2       ) == 4);
   static_assert(LglsOverlapExtents(v2x2,        v4x2       ) == 4);

   static_assert(LglsOverlapExtents(v4x2,        scalar     ) == 8);
   static_assert(LglsOverlapExtents(v4x2,        scalarArray) == 8);
   static_assert(LglsOverlapExtents(v4x2,        smallArray ) == 2);
   static_assert(LglsOverlapExtents(v4x2,        bigArray   ) == 4);
   static_assert(LglsOverlapExtents(v4x2,        v1         ) == 8);
   static_assert(LglsOverlapExtents(v4x2,        v2         ) == 2);
   static_assert(LglsOverlapExtents(v4x2,        v4         ) == 4);
   static_assert(LglsOverlapExtents(v4x2,        v1x2       ) == 2);
   static_assert(LglsOverlapExtents(v4x2,        v2x2       ) == 4);
   static_assert(LglsOverlapExtents(v4x2,        v4x2       ) == 8);
}

TEST_CASE("Lossless checks") {
   static_assert(not CT::Fundamental<Byte>);

   static_assert(Exact<Lossless<std::uint8_t , std::uint8_t >, std::uint8_t >);
   static_assert(Exact<Lossless<std::uint8_t , std::uint16_t>, std::uint16_t>);
   static_assert(Exact<Lossless<std::uint8_t , std::uint32_t>, std::uint32_t>);
   static_assert(Exact<Lossless<std::uint8_t , std::uint64_t>, std::uint64_t>);
   static_assert(Exact<Lossless<std::uint8_t , std::int8_t  >, std::int8_t  >);
   static_assert(Exact<Lossless<std::uint8_t , std::int16_t >, std::int16_t >);
   static_assert(Exact<Lossless<std::uint8_t , std::int32_t >, std::int32_t >);
   static_assert(Exact<Lossless<std::uint8_t , std::int64_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::uint8_t , float        >, float        >);
   static_assert(Exact<Lossless<std::uint8_t , double       >, double       >);
   static_assert(Exact<Lossless<std::uint8_t , Byte         >, Byte         >);

   static_assert(Exact<Lossless<std::uint16_t, std::uint8_t >, std::uint16_t>);
   static_assert(Exact<Lossless<std::uint16_t, std::uint16_t>, std::uint16_t>);
   static_assert(Exact<Lossless<std::uint16_t, std::uint32_t>, std::uint32_t>);
   static_assert(Exact<Lossless<std::uint16_t, std::uint64_t>, std::uint64_t>);
   static_assert(Exact<Lossless<std::uint16_t, std::int8_t  >, std::int16_t >);
   static_assert(Exact<Lossless<std::uint16_t, std::int16_t >, std::int16_t >);
   static_assert(Exact<Lossless<std::uint16_t, std::int32_t >, std::int32_t >);
   static_assert(Exact<Lossless<std::uint16_t, std::int64_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::uint16_t, float        >, float        >);
   static_assert(Exact<Lossless<std::uint16_t, double       >, double       >);
   static_assert(Exact<Lossless<std::uint16_t, Byte         >, Byte         >);

   static_assert(Exact<Lossless<std::uint32_t, std::uint8_t >, std::uint32_t>);
   static_assert(Exact<Lossless<std::uint32_t, std::uint16_t>, std::uint32_t>);
   static_assert(Exact<Lossless<std::uint32_t, std::uint32_t>, std::uint32_t>);
   static_assert(Exact<Lossless<std::uint32_t, std::uint64_t>, std::uint64_t>);
   static_assert(Exact<Lossless<std::uint32_t, std::int8_t  >, std::int32_t >);
   static_assert(Exact<Lossless<std::uint32_t, std::int16_t >, std::int32_t >);
   static_assert(Exact<Lossless<std::uint32_t, std::int32_t >, std::int32_t >);
   static_assert(Exact<Lossless<std::uint32_t, std::int64_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::uint32_t, float        >, float        >);
   static_assert(Exact<Lossless<std::uint32_t, double       >, double       >);
   static_assert(Exact<Lossless<std::uint32_t, Byte         >, Byte         >);

   static_assert(Exact<Lossless<std::uint64_t, std::uint8_t >, std::uint64_t>);
   static_assert(Exact<Lossless<std::uint64_t, std::uint16_t>, std::uint64_t>);
   static_assert(Exact<Lossless<std::uint64_t, std::uint32_t>, std::uint64_t>);
   static_assert(Exact<Lossless<std::uint64_t, std::uint64_t>, std::uint64_t>);
   static_assert(Exact<Lossless<std::uint64_t, std::int8_t  >, std::int64_t >);
   static_assert(Exact<Lossless<std::uint64_t, std::int16_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::uint64_t, std::int32_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::uint64_t, std::int64_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::uint64_t, float        >, float        >);
   static_assert(Exact<Lossless<std::uint64_t, double       >, double       >);
   static_assert(Exact<Lossless<std::uint64_t, Byte         >, Byte         >);


   static_assert(Exact<Lossless<std::int8_t  , std::uint8_t >, std::int8_t  >);
   static_assert(Exact<Lossless<std::int8_t  , std::uint16_t>, std::int16_t >);
   static_assert(Exact<Lossless<std::int8_t  , std::uint32_t>, std::int32_t >);
   static_assert(Exact<Lossless<std::int8_t  , std::uint64_t>, std::int64_t >);
   static_assert(Exact<Lossless<std::int8_t  , std::int8_t  >, std::int8_t  >);
   static_assert(Exact<Lossless<std::int8_t  , std::int16_t >, std::int16_t >);
   static_assert(Exact<Lossless<std::int8_t  , std::int32_t >, std::int32_t >);
   static_assert(Exact<Lossless<std::int8_t  , std::int64_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::int8_t  , float        >, float        >);
   static_assert(Exact<Lossless<std::int8_t  , double       >, double       >);
   static_assert(Exact<Lossless<std::int8_t  , Byte         >, Byte         >);

   static_assert(Exact<Lossless<std::int16_t , std::uint8_t >, std::int16_t >);
   static_assert(Exact<Lossless<std::int16_t , std::uint16_t>, std::int16_t >);
   static_assert(Exact<Lossless<std::int16_t , std::uint32_t>, std::int32_t >);
   static_assert(Exact<Lossless<std::int16_t , std::uint64_t>, std::int64_t >);
   static_assert(Exact<Lossless<std::int16_t , std::int8_t  >, std::int16_t >);
   static_assert(Exact<Lossless<std::int16_t , std::int16_t >, std::int16_t >);
   static_assert(Exact<Lossless<std::int16_t , std::int32_t >, std::int32_t >);
   static_assert(Exact<Lossless<std::int16_t , std::int64_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::int16_t , float        >, float        >);
   static_assert(Exact<Lossless<std::int16_t , double       >, double       >);
   static_assert(Exact<Lossless<std::int16_t , Byte         >, Byte         >);

   static_assert(Exact<Lossless<std::int32_t , std::uint8_t >, std::int32_t >);
   static_assert(Exact<Lossless<std::int32_t , std::uint16_t>, std::int32_t >);
   static_assert(Exact<Lossless<std::int32_t , std::uint32_t>, std::int32_t >);
   static_assert(Exact<Lossless<std::int32_t , std::uint64_t>, std::int64_t >);
   static_assert(Exact<Lossless<std::int32_t , std::int8_t  >, std::int32_t >);
   static_assert(Exact<Lossless<std::int32_t , std::int16_t >, std::int32_t >);
   static_assert(Exact<Lossless<std::int32_t , std::int32_t >, std::int32_t >);
   static_assert(Exact<Lossless<std::int32_t , std::int64_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::int32_t , float        >, float        >);
   static_assert(Exact<Lossless<std::int32_t , double       >, double       >);
   static_assert(Exact<Lossless<std::int32_t , Byte         >, Byte         >);

   static_assert(Exact<Lossless<std::int64_t , std::uint8_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::int64_t , std::uint16_t>, std::int64_t >);
   static_assert(Exact<Lossless<std::int64_t , std::uint32_t>, std::int64_t >);
   static_assert(Exact<Lossless<std::int64_t , std::uint64_t>, std::int64_t >);
   static_assert(Exact<Lossless<std::int64_t , std::int8_t  >, std::int64_t >);
   static_assert(Exact<Lossless<std::int64_t , std::int16_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::int64_t , std::int32_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::int64_t , std::int64_t >, std::int64_t >);
   static_assert(Exact<Lossless<std::int64_t , float        >, float        >);
   static_assert(Exact<Lossless<std::int64_t , double       >, double       >);
   static_assert(Exact<Lossless<std::int64_t , Byte         >, Byte         >);


   static_assert(Exact<Lossless<float        , std::uint8_t >, float        >);
   static_assert(Exact<Lossless<float        , std::uint16_t>, float        >);
   static_assert(Exact<Lossless<float        , std::uint32_t>, float        >);
   static_assert(Exact<Lossless<float        , std::uint64_t>, float        >);
   static_assert(Exact<Lossless<float        , std::int8_t  >, float        >);
   static_assert(Exact<Lossless<float        , std::int16_t >, float        >);
   static_assert(Exact<Lossless<float        , std::int32_t >, float        >);
   static_assert(Exact<Lossless<float        , std::int64_t >, float        >);
   static_assert(Exact<Lossless<float        , float        >, float        >);
   static_assert(Exact<Lossless<float        , double       >, double       >);
   static_assert(Exact<Lossless<float        , Byte         >, Byte         >);


   static_assert(Exact<Lossless<double       , std::uint8_t >, double       >);
   static_assert(Exact<Lossless<double       , std::uint16_t>, double       >);
   static_assert(Exact<Lossless<double       , std::uint32_t>, double       >);
   static_assert(Exact<Lossless<double       , std::uint64_t>, double       >);
   static_assert(Exact<Lossless<double       , std::int8_t  >, double       >);
   static_assert(Exact<Lossless<double       , std::int16_t >, double       >);
   static_assert(Exact<Lossless<double       , std::int32_t >, double       >);
   static_assert(Exact<Lossless<double       , std::int64_t >, double       >);
   static_assert(Exact<Lossless<double       , float        >, double       >);
   static_assert(Exact<Lossless<double       , double       >, double       >);
   static_assert(Exact<Lossless<double       , Byte         >, Byte         >);
}