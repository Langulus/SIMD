///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "Common.hpp"


TEMPLATE_TEST_CASE("Pack 64 bits", "[pack]", ::std::int64_t, ::std::uint64_t) {
   using T = TestType;

#if LANGULUS_SIMD(128BIT)
   GIVEN("A 128bit sequence of numbers") {
      const T n[2] {4, 8};
      const auto r = SIMD::Load<0>(n);

      WHEN("Packed once") {
         const auto r_packed = r.Pack();

         if constexpr (CT::Signed<T>) {
            ::std::int32_t result[2];
            SIMD::Store(r_packed, result);

            for (signed i = 0; i < 2; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
         else {
            ::std::uint32_t result[2];
            SIMD::Store(r_packed, result);

            for (unsigned i = 0; i < 2; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
      }

      WHEN("Packed twice") {
         const auto r_packed = r.Pack().Pack();

         if constexpr (CT::Signed<T>) {
            ::std::int16_t result[2];
            SIMD::Store(r_packed, result);

            for (signed i = 0; i < 2; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
         else {
            ::std::uint16_t result[2];
            SIMD::Store(r_packed, result);

            for (unsigned i = 0; i < 2; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
      }

      WHEN("Packed thrice") {
         const auto r_packed = r.Pack().Pack().Pack();

         if constexpr (CT::Signed<T>) {
            ::std::int8_t result[2];
            SIMD::Store(r_packed, result);

            for (signed i = 0; i < 2; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
         else {
            ::std::uint8_t result[2];
            SIMD::Store(r_packed, result);

            for (unsigned i = 0; i < 2; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
      }
   }
#endif

#if LANGULUS_SIMD(256BIT)
   GIVEN("A 256bit sequence of numbers") {
      const T n[4] {4, 8, 12, 16};
      const auto r = SIMD::Load<0>(n);

      WHEN("Packed once") {
         const auto r_packed = r.Pack();

         if constexpr (CT::Signed<T>) {
            ::std::int32_t result[4];
            SIMD::Store(r_packed, result);

            for (signed i = 0; i < 4; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
         else {
            ::std::uint32_t result[4];
            SIMD::Store(r_packed, result);

            for (unsigned i = 0; i < 4; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
      }

      WHEN("Packed twice") {
         const auto r_packed = r.Pack().Pack();

         if constexpr (CT::Signed<T>) {
            ::std::int16_t result[4];
            SIMD::Store(r_packed, result);

            for (signed i = 0; i < 4; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
         else {
            ::std::uint16_t result[4];
            SIMD::Store(r_packed, result);

            for (unsigned i = 0; i < 4; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
      }

      WHEN("Packed thrice") {
         const auto r_packed = r.Pack().Pack().Pack();

         if constexpr (CT::Signed<T>) {
            ::std::int8_t result[4];
            SIMD::Store(r_packed, result);

            for (signed i = 0; i < 4; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
         else {
            ::std::uint8_t result[4];
            SIMD::Store(r_packed, result);

            for (unsigned i = 0; i < 4; ++i)
               REQUIRE(result[i] == (i + 1) * 4);
         }
      }
   }
#endif

#if LANGULUS_SIMD(512BIT)
   TODO();
#endif
}