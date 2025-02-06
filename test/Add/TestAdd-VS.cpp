///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "TestAdd.hpp"


/*TEMPLATE_TEST_CASE("Vector + Scalar", "[add]"
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
   using E = TypeOf<T>;
   static_assert(CountOf<Vector<signed char, 2>> == 2);

   GIVEN("Vector<T,N> + Scalar<T> = Vector<T,N>") {
      T x;
      E y {};
      T r, rCheck;

      if constexpr (not CT::Vector<T>) {
         InitOne(x,  1);
         InitOne(y, -5);
      }
      else InitOne(y, -5);

      WHEN("Added as constexpr (with saturation)") {
         static_assert(SIMD::Add<true>(T {0}, E {5}) == T {CT::Real<TypeOf<T>> ? 1 : 5});
      }

      WHEN("Added as constexpr (without saturation)") {
         static_assert(SIMD::Add<false>(T {0}, E {5}) == static_cast<T>(5));
      }

      WHEN("Added (with saturation)") {
         ControlAdd<true>(x, y, rCheck);
         SIMD::Add<true>(x, y, r);
            
         REQUIRE(r == rCheck);

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Add (control)") (timer meter) {
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
                  ControlAdd(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
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
                     SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
                  else
                     SIMD::Add(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Added (without saturation)") {
         ControlAdd<false>(x, y, rCheck);
         SIMD::Add<false>(x, y, r);
            
         REQUIRE(r == rCheck);

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Add (control)") (timer meter) {
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
                  ControlAdd(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
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
                     SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
                  else
                     SIMD::Add(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Added in reverse (with saturation)") {
         ControlAdd<true>(y, x, rCheck);
         SIMD::Add<true>(y, x, r);

         REQUIRE(r == rCheck);
      }

      WHEN("Added in reverse (without saturation)") {
         ControlAdd<false>(y, x, rCheck);
         SIMD::Add<false>(y, x, r);

         REQUIRE(r == rCheck);
      }
   }
}*/