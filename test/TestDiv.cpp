///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "Common.hpp"


template<class LHS, class RHS, class OUT> LANGULUS(INLINED)
void ControlDiv(const LHS& lhs, const RHS& rhs, OUT& out) {
   if (rhs == RHS {0})
      LANGULUS_THROW(DivisionByZero, "Division by zero");
   out = lhs / rhs;
}

template<class LHS, class RHS, size_t C, class OUT> LANGULUS(INLINED)
void ControlDiv(const Vector<LHS, C>& lhsArray, const Vector<RHS, C>& rhsArray, Vector<OUT, C>& out) {
   auto r = out.mArray;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   const auto lhsEnd = lhs + C;
   while (lhs != lhsEnd)
      ControlDiv(*lhs++, *rhs++, *r++);
}

TEMPLATE_TEST_CASE("Divide", "[divide]"
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

   GIVEN("x / y = r") {
      T x, y;
      T r, rCheck;

      if constexpr (not CT::Vector<T>) {
         InitOne(x, 1);
         InitOne(y, -5);
      }
      else for (int i = 0; i < CountOf<T>; ++i) {
         if (x[i] == 0)
            x[i] = 1;
         if (y[i] == 0)
            y[i] = 1;
      }

      WHEN("Divided") {
         ControlDiv(x, y, rCheck);
         SIMD::Divide(x, y, r);

         REQUIRE(r == rCheck);

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Divide (control)") (timer meter) {
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
                  ControlDiv(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Divide (SIMD)") (timer meter) {
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
                  SIMD::Divide(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Divided in reverse") {
         ControlDiv(y, x, rCheck);
         SIMD::Divide(y, x, r);

         REQUIRE(r == rCheck);
      }

      WHEN("Divided by zero") {
         if constexpr (not CT::Vector<T>)
            InitOne(x, 0);
         else
            x[0] = 0;

         REQUIRE_THROWS(ControlDiv(y, x, rCheck));
         REQUIRE_THROWS(SIMD::Divide(y, x, r));
      }
   }
}