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
      template<CT::Decayed, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported Max(const T&, const T&) noexcept {
         return {};
      }

      /// Select the bigger values via SIMD                                   
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - type of register we're operating with          
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the maxed values                                          
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto Max(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>)
               return simde_mm_max_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)
               return simde_mm_max_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)
               return simde_mm_max_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>)
               return simde_mm_max_epu16(lhs, rhs);
            else if constexpr (CT::SignedInteger32<T>)
               return simde_mm_max_epi32(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>)
               return simde_mm_max_epu32(lhs, rhs);
            else if constexpr (CT::SignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm_max_epi64(lhs, rhs);
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return simde_mm_max_epu64(lhs, rhs);
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)
               return simde_mm_max_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm_max_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>)
               return simde_mm256_max_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)
               return simde_mm256_max_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)
               return simde_mm256_max_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>)
               return simde_mm256_max_epu16(lhs, rhs);
            else if constexpr (CT::SignedInteger32<T>)
               return simde_mm256_max_epi32(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>)
               return simde_mm256_max_epu32(lhs, rhs);
            else if constexpr (CT::SignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm_max_epi64(lhs, rhs);
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm_max_epu64(lhs, rhs);
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)
               return simde_mm256_max_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm256_max_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else
      #endif
            
      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>)
               return simde_mm512_max_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)
               return simde_mm512_max_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)
               return simde_mm512_max_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>)
               return simde_mm512_max_epu16(lhs, rhs);
            else if constexpr (CT::SignedInteger32<T>)
               return simde_mm512_max_epi32(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>)
               return simde_mm512_max_epu32(lhs, rhs);
            else if constexpr (CT::SignedInteger64<T>)
               return simde_mm512_max_epi64(lhs, rhs);
            else if constexpr (CT::UnsignedInteger64<T>)
               return simde_mm512_max_epu64(lhs, rhs);
            else if constexpr (CT::Float<T>)
               return simde_mm512_max_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm512_max_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else
      #endif
            LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner

   
   /// Max numbers                                                            
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return array/scalar                                                 
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   constexpr auto MaxConstexpr(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using DOUT = Decay<TypeOf<OUT>>;

      return Inner::Evaluate2<0, Unsupported, OUT>(
         lhsOrig, rhsOrig, nullptr,
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            return ::std::max(lhs, rhs);
         }
      );
   }

   /// Max numbers                                                            
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   auto MaxDynamic(LHS& lhsOrig, RHS& rhsOrig) noexcept {
      using DOUT = Decay<TypeOf<OUT>>;
      using REGISTER = Inner::Register<LHS, RHS, OUT>;

      return Inner::Evaluate2<0, REGISTER, OUT>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return Inner::Max<DOUT>(lhs, rhs);
         },
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            return ::std::max(lhs, rhs);
         }
      );
   }

   /// Max numbers, and force output to desired place                         
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT> LANGULUS(INLINED)
   constexpr void Max(LHS& lhs, RHS& rhs, OUT& out) noexcept {
      IF_CONSTEXPR() {
         StoreConstexpr(MaxConstexpr<LHS, RHS, OUT>(lhs, rhs), out);
      }
      else Store(MaxDynamic<LHS, RHS, OUT>(lhs, rhs), out);
   }

   /// Max numbers                                                            
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = std::array<Lossless<Decay<TypeOf<LHS>>, Decay<TypeOf<RHS>>>, OverlapCounts<LHS, RHS>()>>
   LANGULUS(INLINED)
   constexpr OUT Max(const LHS& lhs, const RHS& rhs) noexcept {
      OUT out;
      Max(lhs, rhs, out);
      return out;
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
