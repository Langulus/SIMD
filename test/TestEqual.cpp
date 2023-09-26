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

///                                                                           
template<class T>
constexpr auto BooleanEquivalent() noexcept {
   if constexpr (CT::Vector<T>) {
      if constexpr (CT::Sparse<TypeOf<T>>)
         return Vector<bool*, CountOf<T>> {};
      else
         return Vector<bool, CountOf<T>> {};
   }
   else {
      if constexpr (CT::Sparse<T>)
         return (bool*) nullptr;
      else
         return bool {};
   }
}

template<class T>
using BooleanEquivalentTo = decltype(BooleanEquivalent<T>());

template<class T>
using MaskEquivalentTo = SIMD::Bitmask<CountOf<T>>;

/// Compare two scalars and put result in a bit                               
template<CT::Scalar LHS, CT::Scalar RHS>
LANGULUS(INLINED)
void ControlEqualM(const LHS& lhs, const RHS& rhs, SIMD::Bitmask<1>& out) noexcept {
   out = (DenseCast(lhs) == DenseCast(rhs));
}

/// Compare two vectors and put the result in a bitmask vector                
template<CT::Vector LHS, CT::Vector RHS, class OUT = SIMD::Bitmask<CountOf<LHS>>>
LANGULUS(INLINED)
void ControlEqualM(const LHS& lhsArray, const RHS& rhsArray, OUT& out) noexcept requires (CountOf<LHS> == CountOf<RHS>) {
   using T = typename OUT::Type;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   for (T i = 0; i < T {CountOf<LHS>}; i++)
      out |= (static_cast<T>(DenseCast(*lhs++) == DenseCast(*rhs++)) << i);
}

/// Compare two scalars and put result in a boolean									
template<class LHS, class RHS, CT::Bool OUT>
LANGULUS(INLINED)
void ControlEqualV(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
   DenseCast(out) = (DenseCast(lhs) == DenseCast(rhs));
}

/// Compare two vectors and put the result in a vector of bools               
template<CT::Vector LHS, CT::Vector RHS, CT::Bool OUT>
LANGULUS(INLINED)
void ControlEqualV(const LHS& lhsArray, const RHS& rhsArray, Vector<OUT, CountOf<LHS>>& out) noexcept requires (CountOf<LHS> == CountOf<RHS>) {
   auto r = out.mArray;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   const auto lhsEnd = lhs + CountOf<LHS>;
   while (lhs != lhsEnd)
      ControlEqualV(*lhs++, *rhs++, *r++);
}

///																									
TEMPLATE_TEST_CASE("Compare equality", "[compare]", (Vector<::std::int64_t, 3>)
   /*, NUMBERS_ALL()
   , VECTORS_ALL(1)
   , VECTORS_ALL(2)
   , VECTORS_ALL(3)
   , VECTORS_ALL(4)
   , VECTORS_ALL(5)
   , VECTORS_ALL(8)
   , VECTORS_ALL(9)*/
) {
   using T = TestType;

   GIVEN("x == y = bools") {
      T x, y;
      BooleanEquivalentTo<T> r, rCheck;

      if constexpr (not CT::Vector<T>) {
         if constexpr (CT::Sparse<T>) {
            x = nullptr;
            y = nullptr;
            r = new Decay<BooleanEquivalentTo<T>>;
            rCheck = new Decay<BooleanEquivalentTo<T>>;
         }

         InitOne(x, 1);
         InitOne(y, -5);
      }

      WHEN("Compared for equality as booleans, when guaranteed to be the same") {
         DenseCast(x) = DenseCast(y);
         ControlEqualV(x, y, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Equals(x.mArray, y.mArray, r.mArray);
         else
            SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
            if constexpr (not CT::Vector<T>)
               REQUIRE(DenseCast(r) == true);
            else for (auto it : r)
               REQUIRE(DenseCast(it) == true);
         }
      }

      WHEN("Compared for equality as booleans, when guaranteed to be different") {
         DenseCast(x) = DenseCast(y);
         if constexpr (CT::Vector<T>)
            SIMD::Add(x.mArray, 1, y.mArray);
         else
            SIMD::Add(x, 1, y);

         ControlEqualV(x, y, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Equals(x.mArray, y.mArray, r.mArray);
         else
            SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
            if constexpr (not CT::Vector<T>)
               REQUIRE(DenseCast(r) == false);
            else for (auto it : r)
               REQUIRE(DenseCast(it) == false);
         }
      }

      WHEN("Compared for equality as booleans") {
         ControlEqualV(x, y, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Equals(x.mArray, y.mArray, r.mArray);
         else
            SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
         }

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Equals as booleans (control)") (timer meter) {
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
                  ControlEqualV(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Equals as booleans (SIMD)") (timer meter) {
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
                     SIMD::Equals(nx[i].mArray, ny[i].mArray, nr[i].mArray);
                  else
                     SIMD::Equals(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Compared for equality in reverse (as booleans)") {
         ControlEqualV(y, x, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Equals(x.mArray, y.mArray, r.mArray);
         else
            SIMD::Equals(y, x, r);

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

   GIVEN("x == y = bitmask") {
      T x, y;
      MaskEquivalentTo<T> r, rCheck;

      if constexpr (not CT::Vector<T>) {
         if constexpr (CT::Sparse<T>) {
            x = nullptr;
            y = nullptr;
         }

         InitOne(x, 1);
         InitOne(y, -5);
      }

      WHEN("Compared for equality as bitmask, when guaranteed to be the same") {
         DenseCast(x) = DenseCast(y);

         ControlEqualM(x, y, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Equals(x.mArray, y.mArray, r);
         else
            SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
            REQUIRE(static_cast<bool>(DenseCast(r)) == true);
         }
      }

      WHEN("Compared for equality as bitmask, when guaranteed to be different") {
         DenseCast(x) = DenseCast(y);

         if constexpr (CT::Vector<T>)
            SIMD::Add(x.mArray, 1, y.mArray);
         else
            SIMD::Add(x, 1, y);

         ControlEqualM(x, y, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Equals(x.mArray, y.mArray, r);
         else
            SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
            REQUIRE(static_cast<bool>(DenseCast(r)) == false);
         }
      }

      WHEN("Compared for equality as bitmask, at random") {
         ControlEqualM(x, y, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Equals(x.mArray, y.mArray, r);
         else
            SIMD::Equals(x, y, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
         }

         #ifdef LANGULUS_STD_BENCHMARK
            BENCHMARK_ADVANCED("Equals as bitmask (control)") (timer meter) {
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
                  ControlEqualM(nx[i], ny[i], nr[i]);
               });
            };

            BENCHMARK_ADVANCED("Equals as bitmask (SIMD)") (timer meter) {
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
                     SIMD::Equals(nx[i].mArray, ny[i].mArray, nr[i]);
                  else
                     SIMD::Equals(nx[i], ny[i], nr[i]);
               });
            };
         #endif
      }

      WHEN("Compared for equality in reverse (as bitmask)") {
         ControlEqualM(y, x, rCheck);

         if constexpr (CT::Vector<T>)
            SIMD::Equals(y.mArray, x.mArray, r);
         else
            SIMD::Equals(y, x, r);

         THEN("The result should be correct") {
            REQUIRE(DenseCast(r) == DenseCast(rCheck));
         }
      }

      if constexpr (CT::Sparse<T>) {
         delete x;
         delete y;
      }
   }
}