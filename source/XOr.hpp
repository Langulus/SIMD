///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
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
      template<CT::Decayed, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported XOrSIMD(const T&, const T&) noexcept {
         return {};
      }

      /// XOr two arrays left using SIMD (shifting in zeroes)                 
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the xor'd elements as a register                          
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto XOrSIMD(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
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

      /// XOr numbers at compile-time, if possible                            
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return array/scalar                                              
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto XOrConstexpr(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;

         return Evaluate2<0, Unsupported, OUT>(
            lhsOrig, rhsOrig, nullptr,
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
               return lhs ^ rhs;
            }
         );
      }
   
      /// XOr numbers and return a register, if possible                      
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto XOr(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;
         using REGISTER = Register<decltype(lhsOrig), decltype(rhsOrig), OUT>;

         return Evaluate2<0, REGISTER, OUT>(
            lhsOrig, rhsOrig,
            [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
               LANGULUS_SIMD_VERBOSE("Xoring (SIMD) as ", NameOf<REGISTER>());
               return XOrSIMD<DOUT>(lhs, rhs);
            },
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
               LANGULUS_SIMD_VERBOSE("Xoring (Fallback) ", lhs, " ^ ", rhs, " (", NameOf<DOUT>(), ")");
               return lhs ^ rhs;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(XOr)

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
