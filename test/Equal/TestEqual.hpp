///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "../Main.hpp"


/// Compare two scalars and put result in a bit                               
template<CT::Scalar LHS, CT::Scalar RHS> LANGULUS(INLINED)
void ControlEqualM(const LHS& lhs, const RHS& rhs, SIMD::Bitmask<1>& out) noexcept {
   out = (lhs == rhs);
}

/// Compare two vectors and put the result in a bitmask vector                
template<CT::Vector LHS, CT::Vector RHS, class OUT = SIMD::Bitmask<CountOf<LHS>>>
LANGULUS(INLINED)
void ControlEqualM(const LHS& lhsArray, const RHS& rhsArray, OUT& out)
noexcept requires (CountOf<LHS> == CountOf<RHS>) {
   using T = typename OUT::Type;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   for (T i = 0; i < T {CountOf<LHS>}; i++)
      out |= (static_cast<T>(*lhs++ == *rhs++) << i);
}

/// Compare two scalars and put result in a boolean									
template<class LHS, class RHS, CT::Bool OUT> LANGULUS(INLINED)
void ControlEqualV(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
   out = (lhs == rhs);
}

/// Compare two vectors and put the result in a vector of bools               
template<CT::Vector LHS, CT::Vector RHS, CT::Bool OUT> LANGULUS(INLINED)
void ControlEqualV(const LHS& lhsArray, const RHS& rhsArray, Vector<OUT, CountOf<LHS>>& out)
noexcept requires (CountOf<LHS> == CountOf<RHS>) {
   auto r = out.mArray;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   const auto lhsEnd = lhs + CountOf<LHS>;
   while (lhs != lhsEnd)
      ControlEqualV(*lhs++, *rhs++, *r++);
}



///                                                                           
template<class T>
constexpr auto BooleanEquivalent() noexcept {
   if constexpr (CT::Vector<T>) {
      if constexpr (CT::Sparse<TypeOf<T>>)
         return Vector<bool*, CountOf<T>> {};
      else
         return Vector<bool,  CountOf<T>> {};
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