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
      template<bool SATURATE> LANGULUS(INLINED)
      constexpr Unsupported SubtractSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Subtract two registers                                              
      ///   @tparam SATURATE - whether to clamp to [min;max]                  
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<bool SATURATE, CT::SIMD R> LANGULUS(INLINED)
      auto SubtractSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;
         
         if constexpr (SATURATE) {
            if constexpr (CT::SIMD128<R>) {
               if      constexpr (CT::SignedInteger8<T>)    return R {simde_mm_subs_epi8    (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger8<T>)  return R {simde_mm_subs_epu8    (lhs, rhs)};
               else if constexpr (CT::SignedInteger16<T>)   return R {simde_mm_subs_epi16   (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger16<T>) return R {simde_mm_subs_epu16   (lhs, rhs)};
               else if constexpr (CT::SignedInteger32<T>)
                  return Unsupported {}; //TODO
               else if constexpr (CT::UnsignedInteger32<T>) {
                  const auto mx = simde_mm_max_epu32(lhs, rhs);
                  return R {simde_mm_sub_epi32(mx, rhs)};
               }
               else if constexpr (CT::SignedInteger64<T>)
                  return R {simde_mm_sub_epi64(lhs, rhs)};
               else if constexpr (CT::UnsignedInteger64<T>) {
                  #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
                     const auto mx = simde_mm_max_epu64(lhs, rhs);
                     return R {simde_mm_sub_epi64(mx, rhs)};
                  #else
                     return Unsupported {};
                  #endif
               }
               else if constexpr (CT::Float<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm_max_ps(simde_mm_min_ps(
                     simde_mm_sub_ps(lhs, rhs), simde_mm_set1_ps(1)), simde_mm_set1_ps(0))};
               }
               else if constexpr (CT::Double<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm_max_pd(simde_mm_min_pd(
                     simde_mm_sub_pd(lhs, rhs), simde_mm_set1_pd(1)), simde_mm_set1_pd(0))};
               }
               else static_assert(false, "Unsupported type for 16-byte package");
            }
            else if constexpr (CT::SIMD256<R>) {
               if      constexpr (CT::SignedInteger8<T>)    return R {simde_mm256_subs_epi8 (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger8<T>)  return R {simde_mm256_subs_epu8 (lhs, rhs)};
               else if constexpr (CT::SignedInteger16<T>)   return R {simde_mm256_subs_epi16(lhs, rhs)};
               else if constexpr (CT::UnsignedInteger16<T>) return R {simde_mm256_subs_epu16(lhs, rhs)};
               else if constexpr (CT::SignedInteger32<T>)
                  return Unsupported {}; //TODO
               else if constexpr (CT::UnsignedInteger32<T>) {
                  const auto mx = simde_mm256_max_epu32(lhs, rhs);
                  return R {simde_mm256_sub_epi32(mx, rhs)};
               }
               else if constexpr (CT::SignedInteger64<T>)
                  return R {simde_mm256_sub_epi64(lhs, rhs)};
               else if constexpr (CT::UnsignedInteger64<T>) {
                  #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
                     const auto mx = simde_mm256_max_epu64(lhs, rhs);
                     return R {simde_mm256_sub_epi64(mx, rhs)};
                  #else
                     return Unsupported {};
                  #endif
               }
               else if constexpr (CT::Float<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm256_max_ps(simde_mm256_min_ps(
                     simde_mm256_sub_ps(lhs, rhs), simde_mm256_set1_ps(1)), simde_mm256_set1_ps(0))};
               }
               else if constexpr (CT::Double<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm256_max_pd(simde_mm256_min_pd(
                     simde_mm256_sub_pd(lhs, rhs), simde_mm256_set1_pd(1)), simde_mm256_set1_pd(0))};
               }
               else static_assert(false, "Unsupported type for 32-byte package");
            }
            else if constexpr (CT::SIMD512<R>) {
               if      constexpr (CT::SignedInteger8<T>)    return R {simde_mm512_subs_epi8 (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger8<T>)  return R {simde_mm512_subs_epu8 (lhs, rhs)};
               else if constexpr (CT::SignedInteger16<T>)   return R {simde_mm512_subs_epi16(lhs, rhs)};
               else if constexpr (CT::UnsignedInteger16<T>) return R {simde_mm512_subs_epu16(lhs, rhs)};
               else if constexpr (CT::SignedInteger32<T>)
                  return Unsupported {}; //TODO
               else if constexpr (CT::UnsignedInteger32<T>) {
                  const auto mx = simde_mm512_max_epu32(lhs, rhs);
                  return R {simde_mm512_sub_epi32(mx, rhs)};
               }
               else if constexpr (CT::SignedInteger64<T>)
                  return R {simde_mm512_sub_epi64(lhs, rhs)};
               else if constexpr (CT::UnsignedInteger64<T>) {
                  const auto mx = simde_mm512_max_epu64(lhs, rhs);
                  return R {simde_mm512_sub_epi64(mx, rhs)};
               }
               else if constexpr (CT::Float<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm512_max_ps(simde_mm512_min_ps(
                     simde_mm512_sub_ps(lhs, rhs), simde_mm512_set1_ps(1)), simde_mm512_set1_ps(0))};
               }
               else if constexpr (CT::Double<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm512_max_pd(simde_mm512_min_pd(
                     simde_mm512_sub_pd(lhs, rhs), simde_mm512_set1_pd(1)), simde_mm512_set1_pd(0))};
               }
               else static_assert(false, "Unsupported type for 64-byte package");
            }
            else static_assert(false, "Unsupported type");
         }
         else {
            if constexpr (CT::SIMD128<R>) {
               if      constexpr (CT::Integer8<T>)    return R {simde_mm_sub_epi8    (lhs, rhs)};
               else if constexpr (CT::Integer16<T>)   return R {simde_mm_sub_epi16   (lhs, rhs)};
               else if constexpr (CT::Integer32<T>)   return R {simde_mm_sub_epi32   (lhs, rhs)};
               else if constexpr (CT::Integer64<T>)   return R {simde_mm_sub_epi64   (lhs, rhs)};
               else if constexpr (CT::Float<T>)       return R {simde_mm_sub_ps      (lhs, rhs)};
               else if constexpr (CT::Double<T>)      return R {simde_mm_sub_pd      (lhs, rhs)};
               else static_assert(false, "Unsupported type for 16-byte package");
            }
            else if constexpr (CT::SIMD256<R>) {
               if      constexpr (CT::Integer8<T>)    return R {simde_mm256_sub_epi8 (lhs, rhs)};
               else if constexpr (CT::Integer16<T>)   return R {simde_mm256_sub_epi16(lhs, rhs)};
               else if constexpr (CT::Integer32<T>)   return R {simde_mm256_sub_epi32(lhs, rhs)};
               else if constexpr (CT::Integer64<T>)   return R {simde_mm256_sub_epi64(lhs, rhs)};
               else if constexpr (CT::Float<T>)       return R {simde_mm256_sub_ps   (lhs, rhs)};
               else if constexpr (CT::Double<T>)      return R {simde_mm256_sub_pd   (lhs, rhs)};
               else static_assert(false, "Unsupported type for 32-byte package");
            }
            else if constexpr (CT::SIMD512<R>) {
               if      constexpr (CT::Integer8<T>)    return R {simde_mm512_sub_epi8 (lhs, rhs)};
               else if constexpr (CT::Integer16<T>)   return R {simde_mm512_sub_epi16(lhs, rhs)};
               else if constexpr (CT::Integer32<T>)   return R {simde_mm512_sub_epi32(lhs, rhs)};
               else if constexpr (CT::Integer64<T>)   return R {simde_mm512_sub_epi64(lhs, rhs)};
               else if constexpr (CT::Float<T>)       return R {simde_mm512_sub_ps   (lhs, rhs)};
               else if constexpr (CT::Double<T>)      return R {simde_mm512_sub_pd   (lhs, rhs)};
               else static_assert(false, "Unsupported type for 64-byte package");
            }
            else static_assert(false, "Unsupported type");
         }
      }

      /// Fallback subtraction                                                
      ///   @tparam SATURATE - whether to clamp to max if overflow occurs     
      template<bool SATURATE, class E> LANGULUS(INLINED)
      constexpr E SubtractFallback(const E& lhs, const E& rhs) noexcept {
         if constexpr (SATURATE) {
            using WIDER = WiderSigned<E>;

            if constexpr (sizeof(WIDER) == sizeof(E) and CT::Integer<E>) {
               // If WIDER type isn't wider, perform the saturation     
               // by hand                                               
               constexpr E low = ::std::numeric_limits<E>::min();
               constexpr E hi  = ::std::numeric_limits<E>::max();

               if (rhs > 0)
                  return lhs < low + rhs ? low : lhs - rhs;
               else
                  return lhs > hi  + rhs ? hi  : lhs - rhs;
            }
            else if constexpr (CT::Integer<E>)
               return Saturate<E>(static_cast<WIDER>(lhs) - static_cast<WIDER>(rhs));
            else
               return Saturate<E>(lhs - rhs);
         }
         else return lhs - rhs;
      }

      /// Get difference of values as constexpr, if possible                  
      ///   @tparam SATURATE - whether to clamp to max if overflow occurs     
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the difference scalar/vector                              
      template<bool SATURATE = false, CT::NoIntent FORCE_OUT = void> LANGULUS(INLINED)
      constexpr auto SubtractConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> E {
               return SubtractFallback<SATURATE>(l, r);
            }
         );
      }
   
      /// Get difference of values as a register, if possible                 
      ///   @tparam SATURATE - whether to clamp to max if overflow occurs     
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the difference scalar/vector/register                     
      template<bool SATURATE = false, CT::NoIntent FORCE_OUT = void> LANGULUS(INLINED)
      constexpr auto Subtract(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Subtracting (SIMD) as ", NameOf<R>());
               return SubtractSIMD<SATURATE>(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> E {
               LANGULUS_SIMD_VERBOSE("Subtracting (Fallback) ", l, " - ", r, " (", NameOf<E>(), ")");
               return SubtractFallback<SATURATE>(l, r);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_WITH_SATURATION_API(Subtract)

} // namespace Langulus::SIMD