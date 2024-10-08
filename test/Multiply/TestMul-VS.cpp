///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "TestMul.hpp"


TEMPLATE_TEST_CASE("Vector * Scalar", "[multiply]"
   , VECTORS_ALL(2)
   , NUMBERS_ALL()
   , VECTORS_ALL(1)
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
   using E = TypeOf<T>;
   static_assert(CountOf<Vector<signed char, 2>> == 2);

   GIVEN("Vector<T,N> * Scalar<T> = Vector<T,N>") {
      T x;
      E y {};
      T r, rCheck;

      if constexpr (not CT::Vector<T>) {
         InitOne(x,  1);
         InitOne(y, -5);
      }
      else InitOne(y, -5);

      WHEN("Multiplied as constexpr") {
         constexpr T lhs = E {0};
         constexpr T rhs = E {5};
         constexpr T res = E {0};
         static_assert(SIMD::Multiply(lhs, rhs) == res);
      }

      WHEN("Multiplied") {
         ControlMul(x, y, rCheck);
         SIMD::Multiply(x, y, r);
            
         REQUIRE(r == rCheck);

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Multiply (control)") (timer meter) {
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
                  ControlMul(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Multiply (SIMD)") (timer meter) {
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
                     SIMD::Multiply(nx[i].mArray, ny[i].mArray, nr[i].mArray);
                  else
                     SIMD::Multiply(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Multiplied in reverse") {
         ControlMul(y, x, rCheck);
         SIMD::Multiply(y, x, r);

         REQUIRE(r == rCheck);
      }
   }
}