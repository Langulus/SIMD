///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#include "Common.hpp"


template<CT::Dense B, CT::Dense E> NOD() LANGULUS(INLINED)
constexpr auto Pow(B base, E exponent) noexcept {
   if (base == B {1})
      return B {1};

   if constexpr (CT::IntegerX<B, E>) {
      if constexpr (CT::Unsigned<B>) {
         B result {1};
         while (exponent != E {0}) {
            if ((exponent & E {1}) != E {0})
               result *= base;
            exponent >>= E {1};
            base *= base;
         }
         return result;
      }
      else if (exponent > 0) {
         B result {1};
         while (exponent != E {0}) {
            result *= base;
            --exponent;
         }
         return result;
      }
      else return B {0};
   }
   else if constexpr (CT::Real<B, E>)
      return ::std::pow(base, exponent);
   else
      LANGULUS_ERROR("T must be a number");
}

template<class LHS, class RHS, class OUT> LANGULUS(INLINED)
void ControlPow(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
   out = Pow(lhs, rhs);
}

template<class LHS, class RHS, size_t C, class OUT> LANGULUS(INLINED)
void ControlPow(const Vector<LHS, C>& lhsArray, const Vector<RHS, C>& rhsArray, Vector<OUT, C>& out) noexcept {
   auto r = out.mArray;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   const auto lhsEnd = lhs + C;
   while (lhs != lhsEnd)
      ControlPow(*lhs++, *rhs++, *r++);
}

TEMPLATE_TEST_CASE("Power", "[power]"
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

   GIVEN("pow(x, y) = r") {
      T x, y;
      T r, rCheck;

      if constexpr (not CT::Vector<T>) {
         InitOne(x, 1);
         InitOne(y, -5);
      }

      WHEN("Raised to a power") {
         ControlPow(x, y, rCheck);
         SIMD::Power(x, y, r);

         REQUIRE(r == rCheck);

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Power (control)") (timer meter) {
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
                  ControlPow(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Power (SIMD)") (timer meter) {
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
                  SIMD::Power(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Raise to a power in reverse") {
         ControlPow(y, x, rCheck);
         SIMD::Power(y, x, r);

         REQUIRE(r == rCheck);
      }
   }
}