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
      template<CT::Decayed, CT::NotSIMD T>
      LANGULUS(INLINED)
      constexpr Unsupported XOr(const T&, const T&) noexcept {
         return {};
      }

      /// XOr two arrays left using SIMD (shifting in zeroes)                 
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the xor'd elements as a register                          
      template<CT::Decayed T, CT::SIMD REGISTER>
      LANGULUS(INLINED)
      auto XOr(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::SIMD128i<REGISTER>)
               return simde_mm_xor_si128(lhs, rhs);
            else if constexpr (CT::SIMD128f<REGISTER>)
               return simde_mm_xor_ps(lhs, rhs);
            else if constexpr (CT::SIMD128d<REGISTER>)
               return simde_mm_xor_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::SIMD256i<REGISTER>)
               return simde_mm256_xor_si256(lhs, rhs);
            else if constexpr (CT::SIMD256f<REGISTER>)
               return simde_mm256_xor_ps(lhs, rhs);
            else if constexpr (CT::SIMD256d<REGISTER>)
               return simde_mm256_xor_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::SIMD512i<REGISTER>)
               return simde_mm512_xor_si512(lhs, rhs);
            else if constexpr (CT::SIMD512f<REGISTER>)
               return simde_mm512_xor_ps(lhs, rhs);
            else if constexpr (CT::SIMD512d<REGISTER>)
               return simde_mm512_xor_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for 64-byte package");
      }
         else
      #endif
            LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Add numbers                                                            
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return array/scalar                                                 
   template<class LHS, class RHS, class OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   constexpr auto XOrConstexpr(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using DOUT = Decay<TypeOf<OUT>>;

      return Inner::Evaluate<0, Unsupported, OUT>(
         lhsOrig, rhsOrig, nullptr,
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            return lhs ^ rhs;
         }
      );
   }

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
      using REGISTER = Inner::Register<LHS, RHS, OUT>;

      return Inner::Evaluate<0, REGISTER, OUT>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return Inner::XOr<DOUT>(lhs, rhs);
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
   constexpr void XOr(LHS& lhs, RHS& rhs, OUT& out) noexcept {
      IF_CONSTEXPR() {
         StoreConstexpr(XOrConstexpr<LHS, RHS, OUT>(lhs, rhs), out);
      }
      else Store(XOr<LHS, RHS, OUT>(lhs, rhs), out);
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
