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
      constexpr Unsupported MinSIMD(const T&, const T&) noexcept {
         return {};
      }

      /// Select the bigger values via SIMD                                   
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - type of register we're operating with          
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the maxed values                                          
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto MinSIMD(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>)
               return simde_mm_min_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)
               return simde_mm_min_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)
               return simde_mm_min_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>)
               return simde_mm_min_epu16(lhs, rhs);
            else if constexpr (CT::SignedInteger32<T>)
               return simde_mm_min_epi32(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>)
               return simde_mm_min_epu32(lhs, rhs);
            else if constexpr (CT::SignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm_min_epi64(lhs, rhs);
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm_min_epu64(lhs, rhs);
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)
               return simde_mm_min_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm_min_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>)
               return simde_mm256_min_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)
               return simde_mm256_min_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)
               return simde_mm256_min_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>)
               return simde_mm256_min_epu16(lhs, rhs);
            else if constexpr (CT::SignedInteger32<T>)
               return simde_mm256_min_epi32(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>)
               return simde_mm256_min_epu32(lhs, rhs);
            else if constexpr (CT::SignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm256_min_epi64(lhs, rhs);
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm256_min_epu64(lhs, rhs);
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)
               return simde_mm256_min_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm256_min_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>)
               return simde_mm512_min_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)
               return simde_mm512_min_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)
               return simde_mm512_min_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>)
               return simde_mm512_min_epu16(lhs, rhs);
            else if constexpr (CT::SignedInteger32<T>)
               return simde_mm512_min_epi32(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>)
               return simde_mm512_min_epu32(lhs, rhs);
            else if constexpr (CT::SignedInteger64<T>)
               return simde_mm512_min_epi64(lhs, rhs);
            else if constexpr (CT::UnsignedInteger64<T>)
               return simde_mm512_min_epu64(lhs, rhs);
            else if constexpr (CT::Float<T>)
               return simde_mm512_min_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm512_min_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else
      #endif
            LANGULUS_ERROR("Unsupported type");
      }

      /// Pick the smallest numbers at compile-time, if possible              
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return array/scalar                                              
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto MinConstexpr(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;

         return Evaluate2<0, Unsupported, OUT>(
            lhsOrig, rhsOrig, nullptr,
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
               return ::std::min(lhs, rhs);
            }
         );
      }
   
      /// Pick the smallest numbers and return a register, if possible        
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto Min(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;
         using REGISTER = Register<decltype(lhsOrig), decltype(rhsOrig), OUT>;

         return Evaluate2<0, REGISTER, OUT>(
            lhsOrig, rhsOrig,
            [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
               LANGULUS_SIMD_VERBOSE("Min (SIMD) as ", NameOf<REGISTER>());
               return MinSIMD<DOUT>(lhs, rhs);
            },
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
               LANGULUS_SIMD_VERBOSE("Min (Fallback) Min(", lhs, ", ", rhs, ") (", NameOf<DOUT>(), ")");
               return ::std::min(lhs, rhs);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(Min)

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
