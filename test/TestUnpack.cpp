///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "Common.hpp"


TEMPLATE_TEST_CASE("Unpack 8 bits", "[unpack]", ::std::int8_t, ::std::uint8_t) {

#if LANGULUS_SIMD(128BIT)
   GIVEN("A 128bit sequence of numbers") {
      using T = TestType;
      const T n[16] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
      const auto r = SIMD::Load<0>(n);

      WHEN("Unpacked low once") {
         const auto r_unpacked = r.UnpackLo();

         if constexpr (CT::Signed<T>) {
            ::std::int16_t result[8];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 8; ++i)
               REQUIRE(result[i] == i + 1);
         }
         else {
            ::std::uint16_t result[8];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 8; ++i)
               REQUIRE(result[i] == i + 1);
         }
      }

      WHEN("Unpacked low twice") {
         const auto r_unpacked = r.UnpackLo().UnpackLo();

         if constexpr (CT::Signed<T>) {
            ::std::int32_t result[4];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 4; ++i)
               REQUIRE(result[i] == i + 1);
         }
         else {
            ::std::uint32_t result[4];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 4; ++i)
               REQUIRE(result[i] == i + 1);
         }
      }

      WHEN("Unpacked low thrice") {
         const auto r_unpacked = r.UnpackLo().UnpackLo().UnpackLo();

         if constexpr (CT::Signed<T>) {
            ::std::int64_t result[2];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 2; ++i)
               REQUIRE(result[i] == i + 1);
         }
         else {
            ::std::uint64_t result[2];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 2; ++i)
               REQUIRE(result[i] == i + 1);
         }
      }

      WHEN("Unpacked high once") {
         const auto r_unpacked = r.UnpackHi();

         if constexpr (CT::Signed<T>) {
            ::std::int16_t result[8];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 8; ++i)
               REQUIRE(result[i] == i + 9);
         }
         else {
            ::std::uint16_t result[8];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 8; ++i)
               REQUIRE(result[i] == i + 9);
         }
      }

      WHEN("Unpacked high twice") {
         const auto r_unpacked = r.UnpackHi().UnpackHi();

         if constexpr (CT::Signed<T>) {
            ::std::int32_t result[4];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 4; ++i)
               REQUIRE(result[i] == i + 13);
         }
         else {
            ::std::uint32_t result[4];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 4; ++i)
               REQUIRE(result[i] == i + 13);
         }
      }

      WHEN("Unpacked high thrice") {
         const auto r_unpacked = r.UnpackHi().UnpackHi().UnpackHi();

         if constexpr (CT::Signed<T>) {
            ::std::int64_t result[2];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 2; ++i)
               REQUIRE(result[i] == i + 15);
         }
         else {
            ::std::uint64_t result[2];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 2; ++i)
               REQUIRE(result[i] == i + 15);
         }
      }
   }
#endif

#if LANGULUS_SIMD(256BIT)
   GIVEN("A 256bit sequence of numbers") {
      using T = TestType;
      const T n[32] {
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
      };
      const auto r = SIMD::Load<0>(n);

      WHEN("Unpacked low once") {
         const auto r_unpacked = r.UnpackLo();

         if constexpr (CT::Signed<T>) {
            ::std::int16_t result[16];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 16; ++i)
               REQUIRE(result[i] == i + 1);
         }
         else {
            ::std::uint16_t result[16];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 16; ++i)
               REQUIRE(result[i] == i + 1);
         }
      }

      WHEN("Unpacked low twice") {
         const auto r_unpacked = r.UnpackLo().UnpackLo();

         if constexpr (CT::Signed<T>) {
            ::std::int32_t result[8];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 8; ++i)
               REQUIRE(result[i] == i + 1);
         }
         else {
            ::std::uint32_t result[8];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 8; ++i)
               REQUIRE(result[i] == i + 1);
         }
      }

      WHEN("Unpacked low thrice") {
         const auto r_unpacked = r.UnpackLo().UnpackLo().UnpackLo();

         if constexpr (CT::Signed<T>) {
            ::std::int64_t result[4];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 4; ++i)
               REQUIRE(result[i] == i + 1);
         }
         else {
            ::std::uint64_t result[4];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 4; ++i)
               REQUIRE(result[i] == i + 1);
         }
      }

      WHEN("Unpacked high once") {
         const auto r_unpacked = r.UnpackHi();

         if constexpr (CT::Signed<T>) {
            ::std::int16_t result[16];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 16; ++i)
               REQUIRE(result[i] == i + 17);
         }
         else {
            ::std::uint16_t result[16];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 16; ++i)
               REQUIRE(result[i] == i + 17);
         }
      }

      WHEN("Unpacked high twice") {
         const auto r_unpacked = r.UnpackHi().UnpackHi();

         if constexpr (CT::Signed<T>) {
            ::std::int32_t result[8];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 8; ++i)
               REQUIRE(result[i] == i + 25);
         }
         else {
            ::std::uint32_t result[8];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 8; ++i)
               REQUIRE(result[i] == i + 25);
         }
      }

      WHEN("Unpacked high thrice") {
         const auto r_unpacked = r.UnpackHi().UnpackHi().UnpackHi();

         if constexpr (CT::Signed<T>) {
            ::std::int64_t result[4];
            SIMD::Store(r_unpacked, result);

            for (signed i = 0; i < 4; ++i)
               REQUIRE(result[i] == i + 29);
         }
         else {
            ::std::uint64_t result[4];
            SIMD::Store(r_unpacked, result);

            for (unsigned i = 0; i < 4; ++i)
               REQUIRE(result[i] == i + 29);
         }
      }
   }
#endif

#if LANGULUS_SIMD(512BIT)
   TODO();
#endif
}