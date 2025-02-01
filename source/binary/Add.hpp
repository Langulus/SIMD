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
      constexpr Unsupported AddSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Add two registers                                                   
      ///   @tparam SATURATE - whether to clamp to [min;max]                  
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<bool SATURATE, CT::SIMD R> LANGULUS(INLINED)
      auto AddSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;

         if constexpr (SATURATE) {
            if constexpr (CT::SIMD128<R>) {
               if      constexpr (CT::SignedInteger8<T>)    return R {simde_mm_adds_epi8     (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger8<T>)  return R {simde_mm_adds_epu8     (lhs, rhs)};
               else if constexpr (CT::SignedInteger16<T>)   return R {simde_mm_adds_epi16    (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger16<T>) return R {simde_mm_adds_epu16    (lhs, rhs)};
               else if constexpr (CT::SignedInteger32<T>) {
                  // https://stackoverflow.com/questions/29498824       
                  const auto int_max  = simde_mm_set1_epi32(::std::numeric_limits<T>::max());
                  const auto res      = simde_mm_add_epi32 (lhs, rhs);
                  const auto sign_bit = simde_mm_srli_epi32(lhs, 31);
                  #if LANGULUS_SIMD(AVX512VL)
                     const auto overflow = simde_mm_ternarylogic_epi32(lhs, rhs, res, 0x42);
                  #else
                     const auto sign_xor = simde_mm_xor_si128(lhs, rhs);
                     const auto overflow = simde_mm_andnot_si128(sign_xor, simde_mm_xor_si128(lhs, res));
                  #endif

                  #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
                     return R {simde_mm_mask_add_epi32(res, simde_mm_movepi32_mask(overflow), int_max, sign_bit)};
                  #else
                     const auto saturated = simde_mm_add_epi32(int_max, sign_bit);
                     return R {simde_mm_castps_si128(simde_mm_blendv_ps(
                        simde_mm_castsi128_ps(res),
                        simde_mm_castsi128_ps(saturated),
                        simde_mm_castsi128_ps(overflow)
                     ))};
                  #endif
               }
               else if constexpr (CT::UnsignedInteger32<T>) {
                  const auto mx = simde_mm_min_epu32(lhs, not rhs);
                  return R {simde_mm_add_epi32(mx, rhs)};
               }
               else if constexpr (CT::SignedInteger64<T>)
                  return R {simde_mm_add_epi64(lhs, rhs)};
               else if constexpr (CT::UnsignedInteger64<T>) {
                  #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
                     const auto mx = simde_mm_min_epu64(lhs, not rhs);
                     return R {simde_mm_add_epi64(mx, rhs)};
                  #else
                     return Unsupported {};
                  #endif
               }
               else if constexpr (CT::Float<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm_max_ps(simde_mm_min_ps(
                     simde_mm_add_ps(lhs, rhs), simde_mm_set1_ps(1)), simde_mm_set1_ps(0))};
               }
               else if constexpr (CT::Double<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm_max_pd(simde_mm_min_pd(
                     simde_mm_add_pd(lhs, rhs), simde_mm_set1_pd(1)), simde_mm_set1_pd(0))};
               }
               else static_assert(false, "Unsupported type for 16-byte package");
            }
            else if constexpr (CT::SIMD256<R>) {
               if      constexpr (CT::SignedInteger8<T>)    return R {simde_mm256_adds_epi8  (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger8<T>)  return R {simde_mm256_adds_epu8  (lhs, rhs)};
               else if constexpr (CT::SignedInteger16<T>)   return R {simde_mm256_adds_epi16 (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger16<T>) return R {simde_mm256_adds_epu16 (lhs, rhs)};
               else if constexpr (CT::SignedInteger32<T>) {
                  // https://stackoverflow.com/questions/29498824       
                  const auto int_max  = simde_mm256_set1_epi32(::std::numeric_limits<T>::max());
                  const auto res      = simde_mm256_add_epi32 (lhs, rhs);
                  const auto sign_bit = simde_mm256_srli_epi32(lhs, 31);
                  #if LANGULUS_SIMD(AVX512VL)
                     const auto overflow = simde_mm256_ternarylogic_epi32(lhs, rhs, res, 0x42);
                  #else
                     const auto sign_xor = simde_mm256_xor_si256(lhs, rhs);
                     const auto overflow = simde_mm256_andnot_si256(sign_xor, simde_mm256_xor_si256(lhs, res));
                  #endif

                  #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
                     return R {simde_mm256_mask_add_epi32(res, simde_mm256_movepi32_mask(overflow), int_max, sign_bit)};
                  #else
                     const auto saturated = simde_mm256_add_epi32(int_max, sign_bit);
                     return R {simde_mm256_castps_si256(simde_mm256_blendv_ps(
                        simde_mm256_castsi256_ps(res),
                        simde_mm256_castsi256_ps(saturated),
                        simde_mm256_castsi256_ps(overflow)
                     ))};
                  #endif
               }
               else if constexpr (CT::UnsignedInteger32<T>) {
                  const auto mx = simde_mm256_min_epu32(lhs, not rhs);
                  return R {simde_mm256_add_epi32(mx, rhs)};
               }
               else if constexpr (CT::SignedInteger64<T>)
                  return R {simde_mm256_add_epi64(lhs, rhs)};
               else if constexpr (CT::UnsignedInteger64<T>) {
                  #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
                     const auto mx = simde_mm256_min_epu64(lhs, not rhs);
                     return R {simde_mm256_add_epi64(mx, rhs)};
                  #else
                     return Unsupported {};
                  #endif
               }
               else if constexpr (CT::Float<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm256_max_ps(simde_mm256_min_ps(
                     simde_mm256_add_ps(lhs, rhs), simde_mm256_set1_ps(1)), simde_mm256_set1_ps(0))};
               }
               else if constexpr (CT::Double<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm256_max_pd(simde_mm256_min_pd(
                     simde_mm256_add_pd(lhs, rhs), simde_mm256_set1_pd(1)), simde_mm256_set1_pd(0))};
               }
               else static_assert(false, "Unsupported type for 32-byte package");
            }
            else if constexpr (CT::SIMD512<R>) {
               if      constexpr (CT::SignedInteger8<T>)    return R {simde_mm512_adds_epi8  (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger8<T>)  return R {simde_mm512_adds_epu8  (lhs, rhs)};
               else if constexpr (CT::SignedInteger16<T>)   return R {simde_mm512_adds_epi16 (lhs, rhs)};
               else if constexpr (CT::UnsignedInteger16<T>) return R {simde_mm512_adds_epu16 (lhs, rhs)};
               else if constexpr (CT::SignedInteger32<T>){
                  // https://stackoverflow.com/questions/29498824       
                  const auto int_max  = simde_mm512_set1_epi32(::std::numeric_limits<T>::max());
                  const auto res      = simde_mm512_add_epi32 (lhs, rhs);
                  const auto sign_bit = simde_mm512_srli_epi32(lhs, 31);
                  const auto overflow = simde_mm512_ternarylogic_epi32(lhs, rhs, res, 0x42);
                  return R {simde_mm512_mask_add_epi32(res, simde_mm512_movepi32_mask(overflow), int_max, sign_bit)};
               }
               else if constexpr (CT::UnsignedInteger32<T>) {
                  const auto mx = simde_mm512_min_epu32(lhs, not rhs);
                  return R {simde_mm512_add_epi32(mx, rhs)};
               }
               else if constexpr (CT::SignedInteger64<T>)
                  return R {simde_mm512_add_epi64(lhs, rhs)};
               else if constexpr (CT::UnsignedInteger64<T>) {
                  const auto mx = simde_mm512_min_epu64(lhs, not rhs);
                  return R {simde_mm512_add_epi64(mx, rhs)};
               }
               else if constexpr (CT::Float<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm512_max_ps(simde_mm512_min_ps(
                     simde_mm512_add_ps(lhs, rhs), simde_mm512_set1_ps(1)), simde_mm512_set1_ps(0))};
               }
               else if constexpr (CT::Double<T>) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm512_max_pd(simde_mm512_min_pd(
                     simde_mm512_add_pd(lhs, rhs), simde_mm512_set1_pd(1)), simde_mm512_set1_pd(0))};
               }
               else static_assert(false, "Unsupported type for 64-byte package");
            }
            else static_assert(false, "Unsupported type");
         }
         else {
            if constexpr (CT::SIMD128<R>) {
               if      constexpr (CT::Integer8<T>)          return R {simde_mm_add_epi8      (lhs, rhs)};
               else if constexpr (CT::Integer16<T>)         return R {simde_mm_add_epi16     (lhs, rhs)};
               else if constexpr (CT::Integer32<T>)         return R {simde_mm_add_epi32     (lhs, rhs)};
               else if constexpr (CT::Integer64<T>)         return R {simde_mm_add_epi64     (lhs, rhs)};
               else if constexpr (CT::Float<T>)             return R {simde_mm_add_ps        (lhs, rhs)};
               else if constexpr (CT::Double<T>)            return R {simde_mm_add_pd        (lhs, rhs)};
               else static_assert(false, "Unsupported type for 16-byte package");
            }
            else if constexpr (CT::SIMD256<R>) {
               if      constexpr (CT::Integer8<T>)          return R {simde_mm256_add_epi8   (lhs, rhs)};
               else if constexpr (CT::Integer16<T>)         return R {simde_mm256_add_epi16  (lhs, rhs)};
               else if constexpr (CT::Integer32<T>)         return R {simde_mm256_add_epi32  (lhs, rhs)};
               else if constexpr (CT::Integer64<T>)         return R {simde_mm256_add_epi64  (lhs, rhs)};
               else if constexpr (CT::Float<T>)             return R {simde_mm256_add_ps     (lhs, rhs)};
               else if constexpr (CT::Double<T>)            return R {simde_mm256_add_pd     (lhs, rhs)};
               else static_assert(false, "Unsupported type for 32-byte package");
            }
            else if constexpr (CT::SIMD512<R>) {
               if      constexpr (CT::Integer8<T>)          return R {simde_mm512_add_epi8   (lhs, rhs)};
               else if constexpr (CT::Integer16<T>)         return R {simde_mm512_add_epi16  (lhs, rhs)};
               else if constexpr (CT::Integer32<T>)         return R {simde_mm512_add_epi32  (lhs, rhs)};
               else if constexpr (CT::Integer64<T>)         return R {simde_mm512_add_epi64  (lhs, rhs)};
               else if constexpr (CT::Float<T>)             return R {simde_mm512_add_ps     (lhs, rhs)};
               else if constexpr (CT::Double<T>)            return R {simde_mm512_add_pd     (lhs, rhs)};
               else static_assert(false, "Unsupported type for 64-byte package");
            }
            else static_assert(false, "Unsupported type");
         }
      }

      /// Fallback addition                                                   
      ///   @tparam SATURATE - whether to clamp to max if overflow occurs     
      template<bool SATURATE, class E> LANGULUS(INLINED)
      constexpr E AddFallback(const E& lhs, const E& rhs) noexcept {
         if constexpr (SATURATE) {
            using WIDER = WiderSigned<E>;

            if constexpr (sizeof(WIDER) == sizeof(E) and CT::Integer<E>) {
               // If WIDER type isn't wider, perform the saturation     
               // by hand                                               
               constexpr E lo = ::std::numeric_limits<E>::min();
               constexpr E hi = ::std::numeric_limits<E>::max();

               if (rhs > 0)
                  return lhs > hi - rhs ? hi : lhs + rhs;
               else
                  return lhs < lo - rhs ? lo : lhs + rhs;
            }
            else if constexpr (CT::Integer<E>)
               return Saturate<E>(static_cast<WIDER>(lhs) + static_cast<WIDER>(rhs));
            else
               return Saturate<E>(lhs + rhs);
         }
         else return lhs + rhs;
      }

      /// Get sum of values as constexpr, if possible                         
      ///   @tparam SATURATE - whether to clamp to max if overflow occurs     
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the summed scalar/vector                                  
      template<bool SATURATE = false, CT::NoIntent FORCE_OUT = void> LANGULUS(INLINED)
      constexpr auto AddConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> E {
               return AddFallback<SATURATE>(l, r);
            }
         );
      }
   
      /// Get summed values as a register, if possible                        
      ///   @tparam SATURATE - whether to clamp to max if overflow occurs     
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the summed scalar/vector/register                         
      template<bool SATURATE = false, CT::NoIntent FORCE_OUT = void> LANGULUS(INLINED)
      auto Add(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Adding (SIMD) as ", NameOf<R>());
               return AddSIMD<SATURATE>(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> E {
               LANGULUS_SIMD_VERBOSE("Adding (Fallback) ", l, " + ", r, " (", NameOf<E>(), ")");
               return AddFallback<SATURATE>(l, r);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_WITH_SATURATION_API(Add)

} // namespace Langulus::SIMD
