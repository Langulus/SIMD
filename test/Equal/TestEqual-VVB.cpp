///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#include "TestEqual.hpp"
#include <catch2/catch.hpp>


///                                                                           
TEMPLATE_TEST_CASE("Vector == Vector -> Bool", "[compare]"
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

   GIVEN("x == y = bools") {
      T x, y;
      BooleanEquivalentTo<T> r, rCheck;

      if constexpr (not CT::Vector<T>) {
         if constexpr (CT::Sparse<T>) {
            x = nullptr;
            y = nullptr;
            r = new Decay<BooleanEquivalentTo<T>>;
            rCheck = new Decay<BooleanEquivalentTo<T>>;
         }

         InitOne(x, 1);
         InitOne(y, -5);
      }

      WHEN("Compared for equality as booleans, when guaranteed to be the same") {
         DenseCast(x) = DenseCast(y);

         ControlEqualV(x, y, rCheck);
         SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
            if constexpr (not CT::Vector<T>)
               REQUIRE(DenseCast(r));
            else for (auto it : r)
               REQUIRE(DenseCast(it));
         }
      }

      WHEN("Compared for equality as booleans, when guaranteed to be different") {
         DenseCast(x) = DenseCast(y);
         SIMD::Add(x, 1, y);

         ControlEqualV(x, y, rCheck);
         SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
            if constexpr (not CT::Vector<T>)
               REQUIRE_FALSE(DenseCast(r));
            else for (auto it : r)
               REQUIRE_FALSE(DenseCast(it));
         }
      }

      WHEN("Compared for equality as booleans") {
         ControlEqualV(x, y, rCheck);
         SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
         }

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Equals as booleans (control)") (timer meter) {
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
                  ControlEqualV(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Equals as booleans (SIMD)") (timer meter) {
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

      WHEN("Compared for equality in reverse (as booleans)") {
         ControlEqualV(y, x, rCheck);
         SIMD::Equals(y, x, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
         }
      }

      if constexpr (CT::Sparse<T>) {
         delete r;
         delete rCheck;
         delete x;
         delete y;
      }
   }
}