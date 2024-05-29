///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "TestEqual.hpp"
#include <catch2/catch.hpp>


///                                                                           
TEMPLATE_TEST_CASE("Vector == Vector -> Bitmask", "[compare]"
   , NUMBERS_ALL()
   , VECTORS_ALL(1)
   , VECTORS_ALL(2)
   , VECTORS_ALL(3)
   , VECTORS_ALL(4)
   , VECTORS_ALL(5)
   , VECTORS_ALL(8)
   , VECTORS_ALL(9)
   , VECTORS_ALL(16)
   , VECTORS_ALL(17)
   , VECTORS_ALL(32)
   , VECTORS_ALL(33)
) {
   using T = TestType;

   GIVEN("x == y = bitmask") {
      T x, y;
      MaskEquivalentTo<T> r, rCheck;

      if constexpr (not CT::Vector<T>) {
         if constexpr (CT::Sparse<T>) {
            x = nullptr;
            y = nullptr;
         }

         InitOne(x, 1);
         InitOne(y, -5);
      }

      WHEN("Compared for equality as bitmask, when guaranteed to be the same") {
         DenseCast(x) = DenseCast(y);

         ControlEqualM(x, y, rCheck);
         SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
            REQUIRE(static_cast<bool>(DenseCast(r)) == true);
         }
      }

      WHEN("Compared for equality as bitmask, when guaranteed to be different") {
         DenseCast(x) = DenseCast(y);
         SIMD::Add(x, 1, y);

         ControlEqualM(x, y, rCheck);
         SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
            REQUIRE(static_cast<bool>(DenseCast(r)) == false);
         }
      }

      WHEN("Compared for equality as bitmask, at random") {
         ControlEqualM(x, y, rCheck);
         SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
         }

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Equals as bitmask (control)") (timer meter) {
               some<T> nx(meter.runs());
               if constexpr (not CT::Vector<T>) {
                  for (auto& i : nx)
                     InitOne(i, 1);
               }

               some<T> ny(meter.runs());
               if constexpr (not CT::Vector<T>) {
                  for (auto& i : ny)
                     InitOne(i, 1);
               }

               some<T> nr(meter.runs());
               meter.measure([&](int i) {
                  ControlEqualM(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Equals as bitmask (SIMD)") (timer meter) {
               some<T> nx(meter.runs());
               if constexpr (not CT::Vector<T>) {
                  for (auto& i : nx)
                     InitOne(i, 1);
               }

               some<T> ny(meter.runs());
               if constexpr (not CT::Vector<T>) {
                  for (auto& i : ny)
                     InitOne(i, 1);
               }

               some<T> nr(meter.runs());
               meter.measure([&](int i) {
                  SIMD::Equals(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Compared for equality in reverse (as bitmask)") {
         ControlEqualM(y, x, rCheck);
         SIMD::Equals(y, x, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
         }
      }

      if constexpr (CT::Sparse<T>) {
         delete x;
         delete y;
      }
   }
}