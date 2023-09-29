///                                                                           
/// Langulus::Math                                                            
/// Copyright (c) 2014 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#include "Main.hpp"
#include <SIMD/SIMD.hpp>
#include <catch2/catch.hpp>

using Vec2u8  = Vector<::std::uint8_t,  2>;
using Vec2u16 = Vector<::std::uint16_t, 2>;
using Vec2u32 = Vector<::std::uint32_t, 2>;
using Vec2u64 = Vector<::std::uint64_t, 2>;

using Vec2i8  = Vector<::std::int8_t,   2>;
using Vec2i16 = Vector<::std::int16_t,  2>;
using Vec2i32 = Vector<::std::int32_t,  2>;
using Vec2i64 = Vector<::std::int64_t,  2>;

using Vec2f = Vector<Float, 2>;
using Vec2d = Vector<Double, 2>;


TEMPLATE_TEST_CASE("Testing vector checks", "[vector]", 
   Vec2u8, Vec2u16, Vec2u32, Vec2u64, 
   Vec2i8, Vec2i16, Vec2i32, Vec2i64, 
   Vec2f, Vec2d
) {
   static_assert(    CT::Vector<TestType>);
   static_assert(not CT::Scalar<TestType>);
   static_assert(CountOf <TestType> == 2);
   static_assert(ExtentOf<TestType> == 1);
}

TEMPLATE_TEST_CASE("Testing scalar checks", "[scalar]", NUMBERS_ALL()) {
   static_assert(not CT::Vector<TestType>);
   static_assert(    CT::Scalar<TestType>);
   static_assert(CountOf <TestType> == 1);
   static_assert(ExtentOf<TestType> == 1);
}

TEMPLATE_TEST_CASE("Padding and alignment checks", "[sizes]", NUMBERS_ALL()) {
   using T = TestType;

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

TEMPLATE_TEST_CASE("CountOf checks", "[CountOf]", NUMBERS_ALL()) {
   using T = TestType;

   static_assert(CountOf<T> == 1);

   static_assert(CountOf<Vector<T, 1>> == 1);
   static_assert(CountOf<Vector<T, 2>> == 2);
   static_assert(CountOf<Vector<T, 3>> == 3);
   static_assert(CountOf<Vector<T, 4>> == 4);

   static_assert(CountOf<Vector<T, 1>[12]> == 12);
   static_assert(CountOf<Vector<T, 1>[12]> == 12);
   static_assert(CountOf<Vector<T, 1>[12]> == 12);
   static_assert(CountOf<Vector<T, 2>[12]> == 24);
   static_assert(CountOf<Vector<T, 2>[12]> == 24);
   static_assert(CountOf<Vector<T, 3>[ 8]> == 24);
}

TEST_CASE("OverlapExtents checks", "[OverlapExtents]") {
   int scalar {};
   int scalarArray[1] {};
   int smallArray[2] {};
   int bigArray[4] {};

   static_assert(OVERLAP_EXTENTS(scalar,      scalar     ) == 1);
   static_assert(OVERLAP_EXTENTS(scalar,      scalarArray) == 1);
   static_assert(OVERLAP_EXTENTS(scalar,      smallArray ) == 2);
   static_assert(OVERLAP_EXTENTS(scalar,      bigArray   ) == 4);

   static_assert(OVERLAP_EXTENTS(scalarArray, scalar     ) == 1);
   static_assert(OVERLAP_EXTENTS(scalarArray, scalarArray) == 1);
   static_assert(OVERLAP_EXTENTS(scalarArray, smallArray ) == 2);
   static_assert(OVERLAP_EXTENTS(scalarArray, bigArray   ) == 4);

   static_assert(OVERLAP_EXTENTS(smallArray,  scalar     ) == 2);
   static_assert(OVERLAP_EXTENTS(smallArray,  scalarArray) == 2);
   static_assert(OVERLAP_EXTENTS(smallArray,  smallArray ) == 2);
   static_assert(OVERLAP_EXTENTS(smallArray,  bigArray   ) == 2);

   static_assert(OVERLAP_EXTENTS(bigArray,    scalar     ) == 4);
   static_assert(OVERLAP_EXTENTS(bigArray,    scalarArray) == 4);
   static_assert(OVERLAP_EXTENTS(bigArray,    smallArray ) == 2);
   static_assert(OVERLAP_EXTENTS(bigArray,    bigArray   ) == 4);
}

TEST_CASE("OverlapCounts checks", "[OverlapCounts]") {
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

   static_assert(OverlapCounts<decltype(scalar),      decltype(scalar)     >() == 1);
   static_assert(OverlapCounts<decltype(scalar),      decltype(scalarArray)>() == 1);
   static_assert(OverlapCounts<decltype(scalar),      decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(scalar),      decltype(bigArray)   >() == 4);
   static_assert(OverlapCounts<decltype(scalar),      decltype(v1)         >() == 1);
   static_assert(OverlapCounts<decltype(scalar),      decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(scalar),      decltype(v4)         >() == 4);
   static_assert(OverlapCounts<decltype(scalar),      decltype(v1x2)       >() == 1*2);
   static_assert(OverlapCounts<decltype(scalar),      decltype(v2x2)       >() == 2*2);
   static_assert(OverlapCounts<decltype(scalar),      decltype(v4x2)       >() == 4*2);

   static_assert(OverlapCounts<decltype(scalarArray), decltype(scalar)     >() == 1);
   static_assert(OverlapCounts<decltype(scalarArray), decltype(scalarArray)>() == 1);
   static_assert(OverlapCounts<decltype(scalarArray), decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(scalarArray), decltype(bigArray)   >() == 4);
   static_assert(OverlapCounts<decltype(scalarArray), decltype(v1)         >() == 1);
   static_assert(OverlapCounts<decltype(scalarArray), decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(scalarArray), decltype(v4)         >() == 4);
   static_assert(OverlapCounts<decltype(scalarArray), decltype(v1x2)       >() == 1*2);
   static_assert(OverlapCounts<decltype(scalarArray), decltype(v2x2)       >() == 2*2);
   static_assert(OverlapCounts<decltype(scalarArray), decltype(v4x2)       >() == 4*2);

   static_assert(OverlapCounts<decltype(smallArray),  decltype(scalar)     >() == 2);
   static_assert(OverlapCounts<decltype(smallArray),  decltype(scalarArray)>() == 2);
   static_assert(OverlapCounts<decltype(smallArray),  decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(smallArray),  decltype(bigArray)   >() == 2);
   static_assert(OverlapCounts<decltype(smallArray),  decltype(v1)         >() == 2);
   static_assert(OverlapCounts<decltype(smallArray),  decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(smallArray),  decltype(v4)         >() == 2);
   static_assert(OverlapCounts<decltype(smallArray),  decltype(v1x2)       >() == 2);
   static_assert(OverlapCounts<decltype(smallArray),  decltype(v2x2)       >() == 2);
   static_assert(OverlapCounts<decltype(smallArray),  decltype(v4x2)       >() == 2);

   static_assert(OverlapCounts<decltype(bigArray),    decltype(scalar)     >() == 4);
   static_assert(OverlapCounts<decltype(bigArray),    decltype(scalarArray)>() == 4);
   static_assert(OverlapCounts<decltype(bigArray),    decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(bigArray),    decltype(bigArray)   >() == 4);
   static_assert(OverlapCounts<decltype(bigArray),    decltype(v1)         >() == 4);
   static_assert(OverlapCounts<decltype(bigArray),    decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(bigArray),    decltype(v4)         >() == 4);
   static_assert(OverlapCounts<decltype(bigArray),    decltype(v1x2)       >() == 2);
   static_assert(OverlapCounts<decltype(bigArray),    decltype(v2x2)       >() == 4);
   static_assert(OverlapCounts<decltype(bigArray),    decltype(v4x2)       >() == 4);

   static_assert(OverlapCounts<decltype(v1),          decltype(scalar)     >() == 1);
   static_assert(OverlapCounts<decltype(v1),          decltype(scalarArray)>() == 1);
   static_assert(OverlapCounts<decltype(v1),          decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(v1),          decltype(bigArray)   >() == 4);
   static_assert(OverlapCounts<decltype(v1),          decltype(v1)         >() == 1);
   static_assert(OverlapCounts<decltype(v1),          decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(v1),          decltype(v4)         >() == 4);
   static_assert(OverlapCounts<decltype(v1),          decltype(v1x2)       >() == 1*2);
   static_assert(OverlapCounts<decltype(v1),          decltype(v2x2)       >() == 2*2);
   static_assert(OverlapCounts<decltype(v1),          decltype(v4x2)       >() == 4*2);

   static_assert(OverlapCounts<decltype(v2),          decltype(scalar)     >() == 2);
   static_assert(OverlapCounts<decltype(v2),          decltype(scalarArray)>() == 2);
   static_assert(OverlapCounts<decltype(v2),          decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(v2),          decltype(bigArray)   >() == 2);
   static_assert(OverlapCounts<decltype(v2),          decltype(v1)         >() == 2);
   static_assert(OverlapCounts<decltype(v2),          decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(v2),          decltype(v4)         >() == 2);
   static_assert(OverlapCounts<decltype(v2),          decltype(v1x2)       >() == 2);
   static_assert(OverlapCounts<decltype(v2),          decltype(v2x2)       >() == 2);
   static_assert(OverlapCounts<decltype(v2),          decltype(v4x2)       >() == 2);

   static_assert(OverlapCounts<decltype(v4),          decltype(scalar)     >() == 4);
   static_assert(OverlapCounts<decltype(v4),          decltype(scalarArray)>() == 4);
   static_assert(OverlapCounts<decltype(v4),          decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(v4),          decltype(bigArray)   >() == 4);
   static_assert(OverlapCounts<decltype(v4),          decltype(v1)         >() == 4);
   static_assert(OverlapCounts<decltype(v4),          decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(v4),          decltype(v4)         >() == 4);
   static_assert(OverlapCounts<decltype(v4),          decltype(v1x2)       >() == 2);
   static_assert(OverlapCounts<decltype(v4),          decltype(v2x2)       >() == 4);
   static_assert(OverlapCounts<decltype(v4),          decltype(v4x2)       >() == 4);

   static_assert(OverlapCounts<decltype(v1x2),        decltype(scalar)     >() == 2);
   static_assert(OverlapCounts<decltype(v1x2),        decltype(scalarArray)>() == 2);
   static_assert(OverlapCounts<decltype(v1x2),        decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(v1x2),        decltype(bigArray)   >() == 2);
   static_assert(OverlapCounts<decltype(v1x2),        decltype(v1)         >() == 2);
   static_assert(OverlapCounts<decltype(v1x2),        decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(v1x2),        decltype(v4)         >() == 2);
   static_assert(OverlapCounts<decltype(v1x2),        decltype(v1x2)       >() == 2);
   static_assert(OverlapCounts<decltype(v1x2),        decltype(v2x2)       >() == 2);
   static_assert(OverlapCounts<decltype(v1x2),        decltype(v4x2)       >() == 2);

   static_assert(OverlapCounts<decltype(v2x2),        decltype(scalar)     >() == 4);
   static_assert(OverlapCounts<decltype(v2x2),        decltype(scalarArray)>() == 4);
   static_assert(OverlapCounts<decltype(v2x2),        decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(v2x2),        decltype(bigArray)   >() == 4);
   static_assert(OverlapCounts<decltype(v2x2),        decltype(v1)         >() == 4);
   static_assert(OverlapCounts<decltype(v2x2),        decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(v2x2),        decltype(v4)         >() == 4);
   static_assert(OverlapCounts<decltype(v2x2),        decltype(v1x2)       >() == 2);
   static_assert(OverlapCounts<decltype(v2x2),        decltype(v2x2)       >() == 4);
   static_assert(OverlapCounts<decltype(v2x2),        decltype(v4x2)       >() == 4);

   static_assert(OverlapCounts<decltype(v4x2),        decltype(scalar)     >() == 8);
   static_assert(OverlapCounts<decltype(v4x2),        decltype(scalarArray)>() == 8);
   static_assert(OverlapCounts<decltype(v4x2),        decltype(smallArray) >() == 2);
   static_assert(OverlapCounts<decltype(v4x2),        decltype(bigArray)   >() == 4);
   static_assert(OverlapCounts<decltype(v4x2),        decltype(v1)         >() == 8);
   static_assert(OverlapCounts<decltype(v4x2),        decltype(v2)         >() == 2);
   static_assert(OverlapCounts<decltype(v4x2),        decltype(v4)         >() == 4);
   static_assert(OverlapCounts<decltype(v4x2),        decltype(v1x2)       >() == 2);
   static_assert(OverlapCounts<decltype(v4x2),        decltype(v2x2)       >() == 4);
   static_assert(OverlapCounts<decltype(v4x2),        decltype(v4x2)       >() == 8);
}

TEST_CASE("Lossless checks", "[lossless]") {
   static_assert(not CT::Fundamental<Byte>);

   static_assert(CT::Exact<Lossless<std::uint8_t , std::uint8_t >, std::uint8_t >);
   static_assert(CT::Exact<Lossless<std::uint8_t , std::uint16_t>, std::uint16_t>);
   static_assert(CT::Exact<Lossless<std::uint8_t , std::uint32_t>, std::uint32_t>);
   static_assert(CT::Exact<Lossless<std::uint8_t , std::uint64_t>, std::uint64_t>);
   static_assert(CT::Exact<Lossless<std::uint8_t , std::int8_t  >, std::int8_t  >);
   static_assert(CT::Exact<Lossless<std::uint8_t , std::int16_t >, std::int16_t >);
   static_assert(CT::Exact<Lossless<std::uint8_t , std::int32_t >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::uint8_t , std::int64_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::uint8_t , float        >, float        >);
   static_assert(CT::Exact<Lossless<std::uint8_t , double       >, double       >);
   static_assert(CT::Exact<Lossless<std::uint8_t , Byte         >, std::uint8_t >);

   static_assert(CT::Exact<Lossless<std::uint16_t, std::uint8_t >, std::uint16_t>);
   static_assert(CT::Exact<Lossless<std::uint16_t, std::uint16_t>, std::uint16_t>);
   static_assert(CT::Exact<Lossless<std::uint16_t, std::uint32_t>, std::uint32_t>);
   static_assert(CT::Exact<Lossless<std::uint16_t, std::uint64_t>, std::uint64_t>);
   static_assert(CT::Exact<Lossless<std::uint16_t, std::int8_t  >, std::int16_t >);
   static_assert(CT::Exact<Lossless<std::uint16_t, std::int16_t >, std::int16_t >);
   static_assert(CT::Exact<Lossless<std::uint16_t, std::int32_t >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::uint16_t, std::int64_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::uint16_t, float        >, float        >);
   static_assert(CT::Exact<Lossless<std::uint16_t, double       >, double       >);
   static_assert(CT::Exact<Lossless<std::uint16_t, Byte         >, std::uint16_t>);

   static_assert(CT::Exact<Lossless<std::uint32_t, std::uint8_t >, std::uint32_t>);
   static_assert(CT::Exact<Lossless<std::uint32_t, std::uint16_t>, std::uint32_t>);
   static_assert(CT::Exact<Lossless<std::uint32_t, std::uint32_t>, std::uint32_t>);
   static_assert(CT::Exact<Lossless<std::uint32_t, std::uint64_t>, std::uint64_t>);
   static_assert(CT::Exact<Lossless<std::uint32_t, std::int8_t  >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::uint32_t, std::int16_t >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::uint32_t, std::int32_t >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::uint32_t, std::int64_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::uint32_t, float        >, float        >);
   static_assert(CT::Exact<Lossless<std::uint32_t, double       >, double       >);
   static_assert(CT::Exact<Lossless<std::uint32_t, Byte         >, std::uint32_t>);

   static_assert(CT::Exact<Lossless<std::uint64_t, std::uint8_t >, std::uint64_t>);
   static_assert(CT::Exact<Lossless<std::uint64_t, std::uint16_t>, std::uint64_t>);
   static_assert(CT::Exact<Lossless<std::uint64_t, std::uint32_t>, std::uint64_t>);
   static_assert(CT::Exact<Lossless<std::uint64_t, std::uint64_t>, std::uint64_t>);
   static_assert(CT::Exact<Lossless<std::uint64_t, std::int8_t  >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::uint64_t, std::int16_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::uint64_t, std::int32_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::uint64_t, std::int64_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::uint64_t, float        >, float        >);
   static_assert(CT::Exact<Lossless<std::uint64_t, double       >, double       >);
   static_assert(CT::Exact<Lossless<std::uint64_t, Byte         >, std::uint64_t>);


   static_assert(CT::Exact<Lossless<std::int8_t  , std::uint8_t >, std::int8_t  >);
   static_assert(CT::Exact<Lossless<std::int8_t  , std::uint16_t>, std::int16_t >);
   static_assert(CT::Exact<Lossless<std::int8_t  , std::uint32_t>, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int8_t  , std::uint64_t>, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int8_t  , std::int8_t  >, std::int8_t  >);
   static_assert(CT::Exact<Lossless<std::int8_t  , std::int16_t >, std::int16_t >);
   static_assert(CT::Exact<Lossless<std::int8_t  , std::int32_t >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int8_t  , std::int64_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int8_t  , float        >, float        >);
   static_assert(CT::Exact<Lossless<std::int8_t  , double       >, double       >);
   static_assert(CT::Exact<Lossless<std::int8_t  , Byte         >, std::int8_t  >);

   static_assert(CT::Exact<Lossless<std::int16_t , std::uint8_t >, std::int16_t >);
   static_assert(CT::Exact<Lossless<std::int16_t , std::uint16_t>, std::int16_t >);
   static_assert(CT::Exact<Lossless<std::int16_t , std::uint32_t>, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int16_t , std::uint64_t>, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int16_t , std::int8_t  >, std::int16_t >);
   static_assert(CT::Exact<Lossless<std::int16_t , std::int16_t >, std::int16_t >);
   static_assert(CT::Exact<Lossless<std::int16_t , std::int32_t >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int16_t , std::int64_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int16_t , float        >, float        >);
   static_assert(CT::Exact<Lossless<std::int16_t , double       >, double       >);
   static_assert(CT::Exact<Lossless<std::int16_t , Byte         >, std::int16_t >);

   static_assert(CT::Exact<Lossless<std::int32_t , std::uint8_t >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int32_t , std::uint16_t>, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int32_t , std::uint32_t>, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int32_t , std::uint64_t>, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int32_t , std::int8_t  >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int32_t , std::int16_t >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int32_t , std::int32_t >, std::int32_t >);
   static_assert(CT::Exact<Lossless<std::int32_t , std::int64_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int32_t , float        >, float        >);
   static_assert(CT::Exact<Lossless<std::int32_t , double       >, double       >);
   static_assert(CT::Exact<Lossless<std::int32_t , Byte         >, std::int32_t >);

   static_assert(CT::Exact<Lossless<std::int64_t , std::uint8_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int64_t , std::uint16_t>, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int64_t , std::uint32_t>, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int64_t , std::uint64_t>, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int64_t , std::int8_t  >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int64_t , std::int16_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int64_t , std::int32_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int64_t , std::int64_t >, std::int64_t >);
   static_assert(CT::Exact<Lossless<std::int64_t , float        >, float        >);
   static_assert(CT::Exact<Lossless<std::int64_t , double       >, double       >);
   static_assert(CT::Exact<Lossless<std::int64_t , Byte         >, std::int64_t >);


   static_assert(CT::Exact<Lossless<float        , std::uint8_t >, float        >);
   static_assert(CT::Exact<Lossless<float        , std::uint16_t>, float        >);
   static_assert(CT::Exact<Lossless<float        , std::uint32_t>, float        >);
   static_assert(CT::Exact<Lossless<float        , std::uint64_t>, float        >);
   static_assert(CT::Exact<Lossless<float        , std::int8_t  >, float        >);
   static_assert(CT::Exact<Lossless<float        , std::int16_t >, float        >);
   static_assert(CT::Exact<Lossless<float        , std::int32_t >, float        >);
   static_assert(CT::Exact<Lossless<float        , std::int64_t >, float        >);
   static_assert(CT::Exact<Lossless<float        , float        >, float        >);
   static_assert(CT::Exact<Lossless<float        , double       >, double       >);
   static_assert(CT::Exact<Lossless<float        , Byte         >, float        >);


   static_assert(CT::Exact<Lossless<double       , std::uint8_t >, double       >);
   static_assert(CT::Exact<Lossless<double       , std::uint16_t>, double       >);
   static_assert(CT::Exact<Lossless<double       , std::uint32_t>, double       >);
   static_assert(CT::Exact<Lossless<double       , std::uint64_t>, double       >);
   static_assert(CT::Exact<Lossless<double       , std::int8_t  >, double       >);
   static_assert(CT::Exact<Lossless<double       , std::int16_t >, double       >);
   static_assert(CT::Exact<Lossless<double       , std::int32_t >, double       >);
   static_assert(CT::Exact<Lossless<double       , std::int64_t >, double       >);
   static_assert(CT::Exact<Lossless<double       , float        >, double       >);
   static_assert(CT::Exact<Lossless<double       , double       >, double       >);
   static_assert(CT::Exact<Lossless<double       , Byte         >, double       >);
}