///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "Common.hpp"


template<class LHS, class RHS, class OUT> LANGULUS(INLINED)
void ControlSR(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
   static_assert(CT::IntegerX<LHS, RHS>, "Can only shift integers");
   // Well defined condition in SIMD calls, that is otherwise           
   // undefined behavior by C++ standard                                
   out = rhs < RHS {sizeof(RHS) * 8} and rhs >= 0
      ? lhs >> rhs : 0;
}

template<class LHS, class RHS, size_t C, class OUT> LANGULUS(INLINED)
void ControlSR(const Vector<LHS, C>& lhsArray, const Vector<RHS, C>& rhsArray, Vector<OUT, C>& out) noexcept {
   static_assert(CT::IntegerX<LHS, RHS>, "Can only shift integers");
   auto r = out.mArray;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   const auto lhsEnd = lhs + C;
   while (lhs != lhsEnd)
      ControlSR(*lhs++, *rhs++, *r++);
}

TEMPLATE_TEST_CASE("Shift right", "[shift]"
   , NUMBERS_INT()
   , VECTORS_INT(1)
   , VECTORS_INT(2)
   , VECTORS_INT(3)
   , VECTORS_INT(4)
   , VECTORS_INT(5)
   , VECTORS_INT(8)
   , VECTORS_INT(9)
   , VECTORS_INT(16)
   , VECTORS_INT(17)
   , VECTORS_INT(32)
   , VECTORS_INT(33)
) {
   using T = TestType;

   GIVEN("x >> y = r") {
      T x, y;
      T r, rCheck;

      if constexpr (not CT::Vector<T>) {
         InitOne(x, 1);
         InitOne(y, -5);
      }

      WHEN("Shifted right") {
         ControlSR(x, y, rCheck);
         SIMD::ShiftRight(x, y, r);

         REQUIRE(r == rCheck);

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Shifted right (control)") (timer meter) {
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
                  ControlSR(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Shifted right (SIMD)") (timer meter) {
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
                  SIMD::ShiftRight(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Shifted right in reverse") {
         ControlSR(y, x, rCheck);
         SIMD::ShiftRight(y, x, r);

         REQUIRE(r == rCheck);
      }
   }
}