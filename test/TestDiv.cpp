///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "Main.hpp"
#include <catch2/catch.hpp>

template<class LHS, class RHS, class OUT>
LANGULUS(INLINED)
void ControlDiv(const LHS& lhs, const RHS& rhs, OUT& out) {
   if (rhs == Decay<RHS> {0})
      LANGULUS_THROW(DivisionByZero, "Division by zero");

   out = lhs / rhs;
}

template<class LHS, class RHS, size_t C, class OUT>
LANGULUS(INLINED)
void ControlDiv(const Vector<LHS, C>& lhsArray, const Vector<RHS, C>& rhsArray, Vector<OUT, C>& out) {
   auto r = out.mArray;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   const auto lhsEnd = lhs + C;
   while (lhs != lhsEnd)
      ControlDiv(*lhs++, *rhs++, *r++);
}

TEMPLATE_TEST_CASE("Divide", "[divide]"
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

   GIVEN("x / y = r") {
      T x, y;
      T r, rCheck;

      if constexpr (not CT::Vector<T>) {
         InitOne(x, 1);
         InitOne(y, -5);
      }

      WHEN("Divided") {
         ControlDiv(x, y, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Divide(x.mArray, y.mArray, r.mArray);
         else
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
                  if constexpr (CT::Vector<T>)
                     SIMD::Divide(nx[i].mArray, ny[i].mArray, nr[i].mArray);
                  else
                     SIMD::Divide(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Divided in reverse") {
         ControlDiv(y, x, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Divide(y.mArray, x.mArray, r.mArray);
         else
            SIMD::Divide(y, x, r);

         REQUIRE(r == rCheck);
      }

      WHEN("Divided by zero") {
         if constexpr (not CT::Vector<T>)
            InitOne(x, 0);
         else
            DenseCast(x.mArray[0]) = {};

         REQUIRE_THROWS(ControlDiv(y, x, rCheck));

         if constexpr (CT::Vector<T>)
            REQUIRE_THROWS(SIMD::Divide(y.mArray, x.mArray, r.mArray));
         else
            REQUIRE_THROWS(SIMD::Divide(y, x, r));
      }
   }
}