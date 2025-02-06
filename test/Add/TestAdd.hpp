///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "../Common.hpp"


/// Scalar + Scalar  (either dense or sparse, wrapped or not)                 
/*template<bool SATURATE, CT::Scalar LHS, CT::Scalar RHS, CT::Scalar OUT> LANGULUS(INLINED)
void ControlAdd(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
   auto& fout = FundamentalCast(out);
   fout = SIMD::Inner::AddFallback<SATURATE>(FundamentalCast(lhs), FundamentalCast(rhs));
}

/// Vector + Vector  (either dense or sparse, wrapped or not)                 
template<bool SATURATE, CT::Vector LHS, CT::Vector RHS, CT::Vector OUT> LANGULUS(INLINED)
void ControlAdd(const LHS& lhsArray, const RHS& rhsArray, OUT& out) noexcept {
   static_assert(LHS::MemberCount == RHS::MemberCount
             and LHS::MemberCount == OUT::MemberCount,
      "Vector sizes must match");

   auto r   = out.mArray;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   const auto lhsEnd = lhs + LHS::MemberCount;
   while (lhs != lhsEnd)
      ControlAdd<SATURATE>(*lhs++, *rhs++, *r++);
}

/// Scalar + Vector  (either dense or sparse, wrapped or not)                 
template<bool SATURATE, CT::Scalar LHS, CT::Vector RHS, CT::Vector OUT> LANGULUS(INLINED)
void ControlAdd(const LHS& lhs, const RHS& rhsArray, OUT& out) noexcept {
   static_assert(RHS::MemberCount == OUT::MemberCount,
      "Vector sizes must match");

   auto r   = out.mArray;
   auto rhs = rhsArray.mArray;
   const auto rhsEnd = rhs + RHS::MemberCount;
   while (rhs != rhsEnd)
      ControlAdd<SATURATE>(lhs, *rhs++, *r++);
}

/// Vector + Scalar  (either dense or sparse, wrapped or not)                 
template<bool SATURATE, CT::Vector LHS, CT::Scalar RHS, CT::Vector OUT> LANGULUS(INLINED)
void ControlAdd(const LHS& lhsArray, const RHS& rhs, OUT& out) noexcept {
   ControlAdd<SATURATE>(rhs, lhsArray, out);
}*/