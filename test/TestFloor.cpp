///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#include "Main.hpp"
#include <catch2/catch.hpp>

using timer = Catch::Benchmark::Chronometer;

template<class T>
using uninitialized = Catch::Benchmark::storage_for<T>;

template<class VAL, class OUT> LANGULUS(INLINED)
void ControlFloor(const VAL& val, OUT& out) noexcept {
   DenseCast(out) = std::floor(DenseCast(val));
}

template<class VAL, size_t C, class OUT> LANGULUS(INLINED)
void ControlFloor(const Vector<VAL, C>& a, Vector<OUT, C>& out) noexcept {
   auto r = out.mArray;
   auto va = a.mArray;
   const auto vaEnd = va + C;
   while (va != vaEnd)
      ControlFloor(*va++, *r++);
}

TEMPLATE_TEST_CASE("Floor", "[floor]"
   , NUMBERS_REAL()
   , VECTORS_REAL(1)
   , VECTORS_REAL(2)
   , VECTORS_REAL(3)
   , VECTORS_REAL(4)
   , VECTORS_REAL(5)
   , VECTORS_REAL(8)
   , VECTORS_REAL(9)
   , VECTORS_REAL(16)
   , VECTORS_REAL(17)
   , VECTORS_REAL(32)
   , VECTORS_REAL(33)
) {
   using T = TestType;

   GIVEN("floor(x) = r") {
      T x;
      T r, rCheck;

      if constexpr (not CT::Vector<T>) {
         if constexpr (CT::Sparse<T>) {
            x = nullptr;
            r = new Decay<T>;
            rCheck = new Decay<T>;
         }

         InitOne(x, 1.2);
      }

      WHEN("Floored as constexpr") {
         {
            constexpr T lhs {-1.0};
            constexpr T res {-1.0};
            static_assert(SIMD::Floor(lhs) == res);
         }

         {
            constexpr T lhs {-0.5};
            constexpr T res {-1.0};
            static_assert(SIMD::Floor(lhs) == res);
         }

         {
            constexpr T lhs {0.0};
            constexpr T res {0.0};
            static_assert(SIMD::Floor(lhs) == res);
         }

         {
            constexpr T lhs {1.0};
            constexpr T res {1.0};
            static_assert(SIMD::Floor(lhs) == res);
         }

         {
            constexpr T lhs {1.2};
            constexpr T res {1.0};
            static_assert(SIMD::Floor(lhs) == res);
         }
         {
            constexpr T lhs {1.5};
            constexpr T res {1.0};
            static_assert(SIMD::Floor(lhs) == res);
         }
         {
            constexpr T lhs {2.0};
            constexpr T res {2.0};
            static_assert(SIMD::Floor(lhs) == res);
         }
      }

      WHEN("Floored") {
         ControlFloor(x, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Floor(x.mArray, r.mArray);
         else
            SIMD::Floor(x, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
         }

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Floor (control)") (timer meter) {
               some<T> nx(meter.runs());
               if constexpr (not CT::Vector<T>) {
                  for (auto& i : nx)
                     InitOne(i, 1);
               }

               some<T> nr(meter.runs());
               meter.measure([&](int i) {
                  ControlFloor(nx[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Floor (SIMD)") (timer meter) {
               some<T> nx(meter.runs());
               if constexpr (not CT::Vector<T>) {
                  for (auto& i : nx)
                     InitOne(i, 1.2);
               }

               some<T> nr(meter.runs());
               meter.measure([&](int i) {
                  if constexpr (CT::Vector<T>)
                     SIMD::Floor(nx[i].mArray, nr[i].mArray);
                  else
                     SIMD::Floor(nx[i], nr[i]);
               });
            };
         #endif
      }

      if constexpr (CT::Sparse<T>) {
         delete r;
         delete rCheck;
         delete x;
      }
   }
}