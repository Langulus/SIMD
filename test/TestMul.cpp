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


template<CT::Scalar LHS, CT::Scalar RHS, CT::Scalar OUT>
LANGULUS(INLINED)
void ControlMul(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
   DenseCast(FundamentalCast(out)) = DenseCast(FundamentalCast(lhs))
                                   * DenseCast(FundamentalCast(rhs));
}

template<CT::Vector LHS, CT::Vector RHS, CT::Vector OUT>
LANGULUS(INLINED)
void ControlMul(const LHS& lhsArray, const RHS& rhsArray, OUT& out) noexcept {
   static_assert(LHS::MemberCount == RHS::MemberCount
             and LHS::MemberCount == OUT::MemberCount,
      "Vector sizes must match"
   );

   auto r = out.mArray;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   const auto lhsEnd = lhs + LHS::MemberCount;
   while (lhs != lhsEnd)
      ControlMul(*lhs++, *rhs++, *r++);
}

template<CT::Scalar LHS, CT::Vector RHS, CT::Vector OUT>
LANGULUS(INLINED)
void ControlMul(const LHS& lhs, const RHS& rhsArray, OUT& out) noexcept {
   static_assert(RHS::MemberCount == OUT::MemberCount,
      "Vector sizes must match"
   );

   auto r = out.mArray;
   auto rhs = rhsArray.mArray;
   const auto rhsEnd = rhs + RHS::MemberCount;
   while (rhs != rhsEnd)
      ControlMul(lhs, *rhs++, *r++);
}

template<CT::Vector LHS, CT::Scalar RHS, CT::Vector OUT>
LANGULUS(INLINED)
void ControlMul(const LHS& lhsArray, const RHS& rhs, OUT& out) noexcept {
   static_assert(LHS::MemberCount == OUT::MemberCount,
      "Vector sizes must match"
   );

   auto r = out.mArray;
   auto lhs = lhsArray.mArray;
   const auto lhsEnd = lhs + LHS::MemberCount;
   while (lhs != lhsEnd)
      ControlMul(*lhs++, rhs, *r++);
}

TEMPLATE_TEST_CASE("Vector * Vector", "[multiply]"
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
         if constexpr (CT::Sparse<T>) {
            x = nullptr;
            y = nullptr;
            r = new Decay<T>;
            rCheck = new Decay<T>;
         }

         InitOne(x, 1);
         InitOne(y, -5);
      }

      WHEN("Multiplied") {
         ControlMul(x, y, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Multiply(x.mArray, y.mArray, r.mArray);
         else
            SIMD::Multiply(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
         }

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

         if constexpr (CT::Vector<T>)
            SIMD::Multiply(y.mArray, x.mArray, r.mArray);
         else
            SIMD::Multiply(y, x, r);

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
