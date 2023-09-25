///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Fill.hpp"
#include "Evaluate.hpp"
#include "IgnoreWarningsPush.inl"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<class T, Count S>
      LANGULUS(INLINED)
      constexpr Unsupported XOr(const Unsupported&, const Unsupported&) noexcept {
         return {};
      }

      /// XOr two arrays left using SIMD (shifting in zeroes)                 
      ///   @tparam T - the type of the array element                         
      ///   @tparam S - the size of the array                                 
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the xor'd elements as a register                          
      template<class T, Count S, class REGISTER>
      LANGULUS(INLINED)
      auto XOr(const REGISTER& lhs, const REGISTER& rhs) noexcept {
         if constexpr (CT::Same<REGISTER,simde__m128i>)
            return simde_mm_xor_si128(lhs, rhs);
         else if constexpr (CT::Same<REGISTER,simde__m128>)
            return simde_mm_xor_ps(lhs, rhs);
         else if constexpr (CT::Same<REGISTER,simde__m128d>)
            return simde_mm_xor_pd(lhs, rhs);
         else if constexpr (CT::Same<REGISTER,simde__m256i>)
            return simde_mm256_xor_si256(lhs, rhs);
         else if constexpr (CT::Same<REGISTER,simde__m256>)
            return simde_mm256_xor_ps(lhs, rhs);
         else if constexpr (CT::Same<REGISTER,simde__m256d>)
            return simde_mm256_xor_pd(lhs, rhs);
         else if constexpr (CT::Same<REGISTER,simde__m512i>)
            return simde_mm512_xor_si512(lhs, rhs);
         else if constexpr (CT::Same<REGISTER,simde__m512>)
            return simde_mm512_xor_ps(lhs, rhs);
         else if constexpr (CT::Same<REGISTER,simde__m512d>)
            return simde_mm512_xor_pd(lhs, rhs);
         else
            LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Subtract numbers                                                       
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<class LHS, class RHS, class OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   auto XOr(LHS& lhsOrig, RHS& rhsOrig) noexcept {
      using DOUT = Decay<TypeOf<OUT>>;
      using REGISTER = Inner::Register<LHS, RHS, DOUT>;
      constexpr auto S = OVERLAP_EXTENTS(lhsOrig, rhsOrig);

      return Inner::Evaluate<0, REGISTER, DOUT>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return Inner::XOr<DOUT, S>(lhs, rhs);
         },
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            return lhs ^ rhs;
         }
      );
   }

   /// Subtract numbers, and force output to desired place                    
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<class LHS, class RHS, class OUT>
   LANGULUS(INLINED)
   void XOr(LHS& lhs, RHS& rhs, OUT& output) noexcept {
      GeneralStore(XOr<LHS, RHS, OUT>(lhs, rhs), output);
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
