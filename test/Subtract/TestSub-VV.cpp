///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "TestSub.hpp"


TEMPLATE_TEST_CASE("Vector - Vector", "[subtract]"
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

   GIVEN("x * y = r") {
      T x, y;
      T r, rCheck;

      if constexpr (not CT::Vector<T>) {
         InitOne(x, 1);
         InitOne(y, -5);
      }

      static_assert(SIMD::Subtract<true>(T {0}, T {5}) == T {CT::Signed<TypeOf<T>> and not CT::Real<TypeOf<T>> ? -5 : 0});
      static_assert(SIMD::Subtract<false>(T {0}, T {5}) == static_cast<T>(-5));

      WHEN("Subtracted (with saturation)") {
         ControlSub<true>(x, y, rCheck);
         SIMD::Subtract<true>(x, y, r);

         REQUIRE(r == rCheck);

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Subtract (control)") (timer meter) {
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
                  ControlSub(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Subtract (SIMD)") (timer meter) {
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
                  if constexpr (CT::Vector<T>)
                     SIMD::Subtract(nx[i].mArray, ny[i].mArray, nr[i].mArray);
                  else
                     SIMD::Subtract(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Subtracted (without saturation)") {
         ControlSub<false>(x, y, rCheck);
         SIMD::Subtract<false>(x, y, r);

         REQUIRE(r == rCheck);

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Subtract (control)") (timer meter) {
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
                  ControlSub(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Subtract (SIMD)") (timer meter) {
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
                  if constexpr (CT::Vector<T>)
                     SIMD::Subtract(nx[i].mArray, ny[i].mArray, nr[i].mArray);
                  else
                     SIMD::Subtract(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Subtracted in reverse (with saturation)") {
         ControlSub<true>(y, x, rCheck);
         SIMD::Subtract<true>(y, x, r);

         REQUIRE(r == rCheck);
      }

      WHEN("Subtracted in reverse (without saturation)") {
         ControlSub<false>(y, x, rCheck);
         SIMD::Subtract<false>(y, x, r);

         REQUIRE(r == rCheck);
      }
   }
}
