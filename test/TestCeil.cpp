///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "Common.hpp"


template<class VAL, class OUT> LANGULUS(INLINED)
void ControlCeil(const VAL& val, OUT& out) noexcept {
   out = std::ceil(val);
}

template<class VAL, size_t C, class OUT> LANGULUS(INLINED)
void ControlCeil(const Vector<VAL, C>& a, Vector<OUT, C>& out) noexcept {
   auto r = out.mArray;
   auto va = a.mArray;
   const auto vaEnd = va + C;
   while (va != vaEnd)
      ControlCeil(*va++, *r++);
}

TEMPLATE_TEST_CASE("Ceil", "[ceil]"
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

   GIVEN("ceil(x) = r") {
      T x;
      T r, rCheck;

      if constexpr (not CT::Vector<T>)
         InitOne(x, 1.2);

      WHEN("Ceiled as constexpr") {
         {
            constexpr T lhs = static_cast<T>(-1.0f);
            constexpr T res = static_cast<T>(-1.0f);
            static_assert(SIMD::Ceil(lhs) == res);
         }

         {
            constexpr T lhs = static_cast<T>(-0.5f);
            constexpr T res = static_cast<T>( 0.0f);
            static_assert(SIMD::Ceil(lhs) == res);
         }

         {
            constexpr T lhs = 0;
            constexpr T res = 0;
            static_assert(SIMD::Ceil(lhs) == res);
         }

         {
            constexpr T lhs = static_cast<T>(1.0f);
            constexpr T res = static_cast<T>(1.0f);
            static_assert(SIMD::Ceil(lhs) == res);
         }

         {
            constexpr T lhs = static_cast<T>(1.2f);
            constexpr T res = static_cast<T>(2.0f);
            static_assert(SIMD::Ceil(lhs) == res);
         }
         {
            constexpr T lhs = static_cast<T>(1.5f);
            constexpr T res = static_cast<T>(2.0f);
            static_assert(SIMD::Ceil(lhs) == res);
         }
         {
            constexpr T lhs = static_cast<T>(2.0f);
            constexpr T res = static_cast<T>(2.0f);
            static_assert(SIMD::Ceil(lhs) == res);
         }
      }

      WHEN("Ceiled") {
         ControlCeil(x, rCheck);
         SIMD::Ceil(x, r);

         REQUIRE(r == rCheck);

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Ceil (control)") (timer meter) {
               some<T> nx(meter.runs());
               if constexpr (not CT::Vector<T>) {
                  for (auto& i : nx)
                     InitOne(i, 1);
               }

               some<T> nr(meter.runs());
               meter.measure([&](int i) {
                  ControlCeil(nx[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Ceil (SIMD)") (timer meter) {
               some<T> nx(meter.runs());
               if constexpr (not CT::Vector<T>) {
                  for (auto& i : nx)
                     InitOne(i, 1.2);
               }

               some<T> nr(meter.runs());
               meter.measure([&](int i) {
                  SIMD::Ceil(nx[i], nr[i]);
               });
            };
         #endif
      }
   }
}