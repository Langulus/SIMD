///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "../Attempt.hpp"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      NOD() LANGULUS(INLINED)
      constexpr Unsupported MinSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Pick the smallest numbers in two registers                          
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      auto MinSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;
         
         if constexpr (CT::SIMD128<R>) {
            if constexpr (CT::SignedInteger8<T>)         return R {simde_mm_min_epi8(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger8<T>)  return R {simde_mm_min_epu8(lhs, rhs)};
            else if constexpr (CT::SignedInteger16<T>)   return R {simde_mm_min_epi16(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger16<T>) return R {simde_mm_min_epu16(lhs, rhs)};
            else if constexpr (CT::SignedInteger32<T>)   return R {simde_mm_min_epi32(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger32<T>) return R {simde_mm_min_epu32(lhs, rhs)};
            else if constexpr (CT::SignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return R {_mm_min_epi64(lhs, rhs)};
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return R {simde_mm_min_epu64(lhs, rhs)};
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)             return R {simde_mm_min_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)            return R {simde_mm_min_pd(lhs, rhs)};
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if constexpr (CT::SignedInteger8<T>)         return R {simde_mm256_min_epi8(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger8<T>)  return R {simde_mm256_min_epu8(lhs, rhs)};
            else if constexpr (CT::SignedInteger16<T>)   return R {simde_mm256_min_epi16(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger16<T>) return R {simde_mm256_min_epu16(lhs, rhs)};
            else if constexpr (CT::SignedInteger32<T>)   return R {simde_mm256_min_epi32(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger32<T>) return R {simde_mm256_min_epu32(lhs, rhs)};
            else if constexpr (CT::SignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return R {_mm_min_epi64(lhs, rhs)};
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return R {_mm_min_epu64(lhs, rhs)};
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)             return R {simde_mm256_min_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)            return R {simde_mm256_min_pd(lhs, rhs)};
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if constexpr (CT::SignedInteger8<T>)         return R {simde_mm512_min_epi8(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger8<T>)  return R {simde_mm512_min_epu8(lhs, rhs)};
            else if constexpr (CT::SignedInteger16<T>)   return R {simde_mm512_min_epi16(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger16<T>) return R {simde_mm512_min_epu16(lhs, rhs)};
            else if constexpr (CT::SignedInteger32<T>)   return R {simde_mm512_min_epi32(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger32<T>) return R {simde_mm512_min_epu32(lhs, rhs)};
            else if constexpr (CT::SignedInteger64<T>)   return R {simde_mm512_min_epi64(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger64<T>) return R {simde_mm512_min_epu64(lhs, rhs)};
            else if constexpr (CT::Float<T>)             return R {simde_mm512_min_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)            return R {simde_mm512_min_pd(lhs, rhs)};
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }
      
      /// Get smallest values as constexpr, if possible                       
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the smallest scalar/vector                                
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto MinConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> E {
               return ::std::min(l, r);
            }
         );
      }
   
      /// Get smallest values as a register, if possible                      
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the smallest scalar/vector/register                       
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Min(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Min (SIMD) as ", NameOf<R>());
               return MinSIMD(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> E {
               LANGULUS_SIMD_VERBOSE("Min (Fallback) ", l, ", ", r, " (", NameOf<E>(), ")");
               return ::std::min(l, r);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(Min)

} // namespace Langulus::SIMD
