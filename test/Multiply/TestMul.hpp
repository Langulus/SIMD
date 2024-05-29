///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "../Main.hpp"


/// Scalar * Scalar  (either dense or sparse, wrapped or not)                 
///   @attention 8bit integers are always multiplied with saturation          
template<CT::Scalar LHS, CT::Scalar RHS, CT::Scalar OUT> LANGULUS(INLINED)
void ControlMul(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
   auto& fout = FundamentalCast(out);
   if constexpr (CT::Same<decltype(fout), uint8_t>) {
      // 8-bit unsigned multiplication with saturation                  
      const unsigned temp = FundamentalCast(lhs) * FundamentalCast(rhs);
      fout = temp > 255 ? 255 : temp;
   }
   else if constexpr (CT::Same<decltype(fout), int8_t>) {
      // 8-bit signed multiplication with saturation                    
      const signed temp = FundamentalCast(lhs) * FundamentalCast(rhs);
      fout = temp > 127 ? 127 : (temp < -128 ? -128 : temp);
   }
   else fout = FundamentalCast(lhs) * FundamentalCast(rhs);
}

/// Vector * Vector  (either dense or sparse, wrapped or not)                 
template<CT::Vector LHS, CT::Vector RHS, CT::Vector OUT> LANGULUS(INLINED)
void ControlMul(const LHS& lhsArray, const RHS& rhsArray, OUT& out) noexcept {
   static_assert(LHS::MemberCount == RHS::MemberCount
             and LHS::MemberCount == OUT::MemberCount,
      "Vector sizes must match");

   auto r   = out.mArray;
   auto lhs = lhsArray.mArray;
   auto rhs = rhsArray.mArray;
   const auto lhsEnd = lhs + LHS::MemberCount;
   while (lhs != lhsEnd)
      ControlMul(*lhs++, *rhs++, *r++);
}

/// Scalar * Vector  (either dense or sparse, wrapped or not)                 
template<CT::Scalar LHS, CT::Vector RHS, CT::Vector OUT> LANGULUS(INLINED)
void ControlMul(const LHS& lhs, const RHS& rhsArray, OUT& out) noexcept {
   static_assert(RHS::MemberCount == OUT::MemberCount,
      "Vector sizes must match");

   auto r   = out.mArray;
   auto rhs = rhsArray.mArray;
   const auto rhsEnd = rhs + RHS::MemberCount;
   while (rhs != rhsEnd)
      ControlMul(lhs, *rhs++, *r++);
}

/// Vector * Scalar  (either dense or sparse, wrapped or not)                 
template<CT::Vector LHS, CT::Scalar RHS, CT::Vector OUT> LANGULUS(INLINED)
void ControlMul(const LHS& lhsArray, const RHS& rhs, OUT& out) noexcept {
   return ControlMul(rhs, lhsArray, out);
}