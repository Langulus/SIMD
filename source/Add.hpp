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
#include "Store.hpp"
#include "IgnoreWarningsPush.inl"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<CT::Decayed, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported AddSIMD(const T&, const T&) noexcept {
         return {};
      }

      /// Add two arrays using SIMD                                           
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the added elements as a register                          
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto AddSIMD(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
         #if LANGULUS_SIMD(128BIT)
            if constexpr (CT::SIMD128<REGISTER>) {
               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm_add_epi8(lhs, rhs);
               else if constexpr (CT::UnsignedInteger8<T>)
                  return simde_mm_adds_epu8(lhs, rhs);
               else if constexpr (CT::SignedInteger16<T>)
                  return simde_mm_add_epi16(lhs, rhs);
               else if constexpr (CT::UnsignedInteger16<T>)
                  return simde_mm_adds_epu16(lhs, rhs);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm_add_epi32(lhs, rhs);
               else if constexpr (CT::Integer64<T>)
                  return simde_mm_add_epi64(lhs, rhs);
               else if constexpr (CT::Float<T>)
                  return simde_mm_add_ps(lhs, rhs);
               else if constexpr (CT::Double<T>)
                  return simde_mm_add_pd(lhs, rhs);
               else
                  LANGULUS_ERROR("Unsupported type for 16-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(256BIT)
            if constexpr (CT::SIMD256<REGISTER>) {
               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm256_add_epi8(lhs, rhs);
               else if constexpr (CT::UnsignedInteger8<T>)
                  return simde_mm256_adds_epu8(lhs, rhs);
               else if constexpr (CT::SignedInteger16<T>)
                  return simde_mm256_add_epi16(lhs, rhs);
               else if constexpr (CT::UnsignedInteger16<T>)
                  return simde_mm256_adds_epu16(lhs, rhs);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm256_add_epi32(lhs, rhs);
               else if constexpr (CT::Integer64<T>)
                  return simde_mm256_add_epi64(lhs, rhs);
               else if constexpr (CT::Float<T>)
                  return simde_mm256_add_ps(lhs, rhs);
               else if constexpr (CT::Double<T>)
                  return simde_mm256_add_pd(lhs, rhs);
               else
                  LANGULUS_ERROR("Unsupported type for 32-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(512BIT)
            if constexpr (CT::SIMD512<REGISTER>) {
               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm512_add_epi8(lhs, rhs);
               else if constexpr (CT::UnsignedInteger8<T>)
                  return simde_mm512_adds_epu8(lhs, rhs);
               else if constexpr (CT::SignedInteger16<T>)
                  return simde_mm512_add_epi16(lhs, rhs);
               else if constexpr (CT::UnsignedInteger16<T>)
                  return simde_mm512_adds_epu16(lhs, rhs);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm512_add_epi32(lhs, rhs);
               else if constexpr (CT::Integer64<T>)
                  return simde_mm512_add_epi64(lhs, rhs);
               else if constexpr (CT::Float<T>)
                  return simde_mm512_add_ps(lhs, rhs);
               else if constexpr (CT::Double<T>)
                  return simde_mm512_add_pd(lhs, rhs);
               else
                  LANGULUS_ERROR("Unsupported type for 64-byte package");
            }
            else
         #endif
            LANGULUS_ERROR("Unsupported type");
      }

      /// Add numbers at compile-time, if possible                            
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return array/scalar                                              
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto AddConstexpr(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;

         return Evaluate2<0, Unsupported, OUT>(
            lhsOrig, rhsOrig, nullptr,
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
               return lhs + rhs;
            }
         );
      }
   
      /// Add numbers and return a register, if possible                      
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto Add(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;
         using REGISTER = Register<decltype(lhsOrig), decltype(rhsOrig), OUT>;

         return Evaluate2<0, REGISTER, OUT>(
            lhsOrig, rhsOrig,
            [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
               LANGULUS_SIMD_VERBOSE("Adding (SIMD) as ", NameOf<REGISTER>());
               return AddSIMD<DOUT>(lhs, rhs);
            },
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
               LANGULUS_SIMD_VERBOSE("Adding (Fallback) ", lhs, " + ", rhs, " (", NameOf<DOUT>(), ")");
               return lhs + rhs;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(Add)

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
