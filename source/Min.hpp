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
#include "Convert.hpp"
#include "IgnoreWarningsPush.inl"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<class T, Count S>
      LANGULUS(INLINED)
      constexpr Unsupported Min(const Unsupported&, const Unsupported&) noexcept {
         return {};
      }

      /// Select the bigger values via SIMD                                   
      ///   @tparam T - the type of the array element                         
      ///   @tparam S - the size of the array                                 
      ///   @tparam REGISTER - type of register we're operating with          
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the maxed values                                          
      template<class T, Count S, CT::TSIMD REGISTER>
      LANGULUS(INLINED)
      auto Min(const REGISTER& lhs, const REGISTER& rhs) noexcept {
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
         else if constexpr (CT::SIMD256<REGISTER>) {
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
         else if constexpr (CT::SIMD512<REGISTER>) {
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
         else LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Min numbers                                                            
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<class LHS, class RHS, class OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   auto Min(LHS& lhsOrig, RHS& rhsOrig) noexcept {
      using DOUT = Decay<OUT>;
      using REGISTER = CT::Register<LHS, RHS, DOUT>;
      constexpr auto S = OverlapCount<LHS, RHS>();

      return Inner::Evaluate<0, REGISTER, DOUT>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return Inner::Min<DOUT, S>(lhs, rhs);
         },
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            return ::std::min(lhs, rhs);
         }
      );
   }

   /// Min numbers, and force output to desired place                         
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<class LHS, class RHS, class OUT>
   LANGULUS(INLINED)
   void Min(LHS& lhs, RHS& rhs, OUT& output) noexcept {
      GeneralStore(Min<LHS, RHS, OUT>(lhs, rhs), output);
   }

   ///                                                                        
   template<CT::Vector WRAPPER, class LHS, class RHS>
   NOD() LANGULUS(INLINED)
   WRAPPER MinWrap(LHS& lhs, RHS& rhs) noexcept {
      WRAPPER result;
      Min<LHS, RHS>(lhs, rhs, result.mArray);
      return result;
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
