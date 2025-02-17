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
      constexpr Unsupported MultiplySIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Multiply two registers                                              
      ///   @tparam SATURATE - whether to clamp integer vectors on overflow   
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<bool SATURATE, CT::SIMD R> LANGULUS(INLINED)
      auto MultiplySIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;

      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<R>) {
            if constexpr (CT::Integer8<T>) {
               auto lhsi16 = CT::Signed<T> ? simde_mm_cvtepi8_epi16(lhs) : simde_mm_cvtepu8_epi16(lhs);
               auto rhsi16 = CT::Signed<T> ? simde_mm_cvtepi8_epi16(rhs) : simde_mm_cvtepu8_epi16(rhs);
               const auto lo = simde_mm_mullo_epi16(lhsi16, rhsi16);
               lhs = _mm_halfflip(lhs);
               rhs = _mm_halfflip(rhs);
                     lhsi16 = CT::Signed<T> ? simde_mm_cvtepi8_epi16(lhs) : simde_mm_cvtepu8_epi16(lhs);
                     rhsi16 = CT::Signed<T> ? simde_mm_cvtepi8_epi16(rhs) : simde_mm_cvtepu8_epi16(rhs);
               const auto hi = simde_mm_mullo_epi16(lhsi16, rhsi16);

               if constexpr (SATURATE) {
                  // Saturation happens via packing                     
                  if constexpr (CT::SignedInteger8<T>)
                     return R {simde_mm_packs_epi16(lo, hi)};
                  else
                     return R {simde_mm_packus_epi16(lo, hi)};
               }
               else {
                  #if LANGULUS_SIMD(AVX512BW) and LANGULUS_SIMD(AVX512VL)
                     return R {simde_mm_cvtepi16_epi8(lo), simde_mm_cvtepi16_epi8(hi)};
                  #else
                     return Unsupported {};
                  #endif
               }
            }
            else if constexpr (CT::Integer16<T>) {
               if constexpr (SATURATE) {
                  auto lhsi32 = CT::Signed<T> ? simde_mm_cvtepi16_epi32(lhs) : simde_mm_cvtepu16_epi32(lhs);
                  auto rhsi32 = CT::Signed<T> ? simde_mm_cvtepi16_epi32(rhs) : simde_mm_cvtepu16_epi32(rhs);
                  const auto lo = simde_mm_mullo_epi32(lhsi32, rhsi32);
                  lhs = _mm_halfflip(lhs);
                  rhs = _mm_halfflip(rhs);
                       lhsi32 = CT::Signed<T> ? simde_mm_cvtepi16_epi32(lhs) : simde_mm_cvtepu16_epi32(lhs);
                       rhsi32 = CT::Signed<T> ? simde_mm_cvtepi16_epi32(rhs) : simde_mm_cvtepu16_epi32(rhs);
                  const auto hi = simde_mm_mullo_epi32(lhsi32, rhsi32);

                  // Saturation happens via packing                     
                  if constexpr (CT::SignedInteger16<T>)
                     return R {simde_mm_packs_epi32(lo, hi)};
                  else
                     return R {simde_mm_packus_epi32(lo, hi)};
               }
               else return R {simde_mm_mullo_epi16(lhs, rhs)};
            }
            else if constexpr (CT::Integer32<T>) {
               if constexpr (SATURATE) {
                  #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL) // for simde_mm_cvtsepi64_epi32
                     auto lhsi64 = CT::Signed<T> ? simde_mm_cvtepi32_epi64(lhs) : simde_mm_cvtepu32_epi64(lhs);
                     auto rhsi64 = CT::Signed<T> ? simde_mm_cvtepi32_epi64(rhs) : simde_mm_cvtepu32_epi64(rhs);
                     const auto lo = CT::Signed<T> ? simde_mm_mul_epi32(lhsi64, rhsi64) : simde_mm_mul_epu32(lhsi64, rhsi64);
                     lhs = _mm_halfflip(lhs);
                     rhs = _mm_halfflip(rhs);
                          lhsi64 = CT::Signed<T> ? simde_mm_cvtepi32_epi64(lhs) : simde_mm_cvtepu32_epi64(lhs);
                          rhsi64 = CT::Signed<T> ? simde_mm_cvtepi32_epi64(rhs) : simde_mm_cvtepu32_epi64(rhs);
                     const auto hi = CT::Signed<T> ? simde_mm_mul_epi32(lhsi64, rhsi64) : simde_mm_mul_epu32(lhsi64, rhsi64);

                     // Saturation happens via packing                  
                     return R {
                        simde_mm_cvtsepi64_epi32(lo),
                        simde_mm_cvtsepi64_epi32(hi)
                     };
                  #else
                     return Unsupported {};
                  #endif
               }
               else return R {simde_mm_mullo_epi32(lhs, rhs)};
            }
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
                  if constexpr (not SATURATE)
                     return R {simde_mm_mullo_epi64(lhs, rhs)};
                  else
                     return Unsupported {};
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>) {
               if constexpr (SATURATE) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm_max_ps(simde_mm_min_ps(
                     simde_mm_mul_ps(lhs, rhs), simde_mm_set1_ps(1)), simde_mm_set1_ps(0))};
               }
               else return R {simde_mm_mul_ps(lhs, rhs)};
            }
            else if constexpr (CT::Double<T>) {
               if constexpr (SATURATE) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm_max_pd(simde_mm_min_pd(
                     simde_mm_mul_pd(lhs, rhs), simde_mm_set1_pd(1)), simde_mm_set1_pd(0))};
               }
               else return R {simde_mm_mul_pd(lhs, rhs)};
            }
            else static_assert(false, "Unsupported type for 16-byte package");
         }
         else
      #endif
      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<R>) {
            if constexpr (CT::Integer8<T>) {
               auto lhsi16 = CT::Signed<T> ? simde_mm256_cvtepi8_epi16(_mm256_castsi256_si128(lhs)) : simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(lhs));
               auto rhsi16 = CT::Signed<T> ? simde_mm256_cvtepi8_epi16(_mm256_castsi256_si128(rhs)) : simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(rhs));
               const auto lo = simde_mm256_mullo_epi16(lhsi16, rhsi16);
               lhs = _mm_halfflip(lhs);
               rhs = _mm_halfflip(rhs);
                     lhsi16 = CT::Signed<T> ? simde_mm256_cvtepi8_epi16(_mm256_castsi256_si128(lhs)) : simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(lhs));
                     rhsi16 = CT::Signed<T> ? simde_mm256_cvtepi8_epi16(_mm256_castsi256_si128(rhs)) : simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(rhs));
               const auto hi = simde_mm256_mullo_epi16(lhsi16, rhsi16);
               return lgls_pack_epi16<SATURATE>(V256<Wider<T>> {lo}, V256<Wider<T>> {hi});
            }
            else if constexpr (CT::Integer16<T>) {
               if constexpr (SATURATE) {
                  auto lhsi32 = CT::Signed<T> ? simde_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lhs)) : simde_mm256_cvtepu16_epi32(_mm256_castsi256_si128(lhs));
                  auto rhsi32 = CT::Signed<T> ? simde_mm256_cvtepi16_epi32(_mm256_castsi256_si128(rhs)) : simde_mm256_cvtepu16_epi32(_mm256_castsi256_si128(rhs));
                  const auto lo = simde_mm256_mullo_epi32(lhsi32, rhsi32);
                  lhs = _mm_halfflip(lhs);
                  rhs = _mm_halfflip(rhs);
                        lhsi32 = CT::Signed<T> ? simde_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lhs)) : simde_mm256_cvtepu16_epi32(_mm256_castsi256_si128(lhs));
                        rhsi32 = CT::Signed<T> ? simde_mm256_cvtepi16_epi32(_mm256_castsi256_si128(rhs)) : simde_mm256_cvtepu16_epi32(_mm256_castsi256_si128(rhs));
                  const auto hi = simde_mm256_mullo_epi32(lhsi32, rhsi32);
                  return lgls_pack_epi32<true>(V256<Wider<T>> {lo}, V256<Wider<T>> {hi});
               }
               else return R {simde_mm256_mullo_epi16(lhs, rhs)};
            }
            else if constexpr (CT::Integer32<T>) {
               if constexpr (SATURATE) {
                  #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL) // for simde_mm256_cvtsepi64_epi32
                     auto lhsi64 = CT::Signed<T> ? simde_mm256_cvtepi32_epi64(lhs) : simde_mm256_cvtepu32_epi64(lhs);
                     auto rhsi64 = CT::Signed<T> ? simde_mm256_cvtepi32_epi64(rhs) : simde_mm256_cvtepu32_epi64(rhs);
                     const auto lo = CT::Signed<T> ? simde_mm256_mul_epi32(lhsi64, rhsi64) : simde_mm256_mul_epu32(lhsi64, rhsi64);
                     lhs = _mm_halfflip(lhs);
                     rhs = _mm_halfflip(rhs);
                          lhsi64 = CT::Signed<T> ? simde_mm256_cvtepi32_epi64(lhs) : simde_mm256_cvtepu32_epi64(lhs);
                          rhsi64 = CT::Signed<T> ? simde_mm256_cvtepi32_epi64(rhs) : simde_mm256_cvtepu32_epi64(rhs);
                     const auto hi = CT::Signed<T> ? simde_mm256_mul_epi32(lhsi64, rhsi64) : simde_mm256_mul_epu32(lhsi64, rhsi64);

                     // Saturation happens via packing                  
                     return R {
                        simde_mm256_cvtsepi64_epi32(lo),
                        simde_mm256_cvtsepi64_epi32(hi)
                     };
                  #else
                     return Unsupported {};
                  #endif
               }
               else return R {simde_mm256_mullo_epi32(lhs, rhs)};
            }
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
                  if constexpr (not SATURATE)
                     return R {simde_mm256_mullo_epi64(lhs, rhs)};
                  else
                     return Unsupported {};
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>) {
               if constexpr (SATURATE) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm256_max_ps(simde_mm256_min_ps(
                     simde_mm256_mul_ps(lhs, rhs), simde_mm256_set1_ps(1)), simde_mm256_set1_ps(0))};
               }
               else return R {simde_mm256_mul_ps(lhs, rhs)};
            }
            else if constexpr (CT::Double<T>) {
               if constexpr (SATURATE) {
                  // Clamp to [0;1] range                               
                  return R {simde_mm256_max_pd(simde_mm256_min_pd(
                     simde_mm256_mul_pd(lhs, rhs), simde_mm256_set1_pd(1)), simde_mm256_set1_pd(0))};
               }
               else return R {simde_mm256_mul_pd(lhs, rhs)};
            }
            else static_assert(false, "Unsupported type for 32-byte package");
         }
         else
      #endif
      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<R>) {
            if constexpr (CT::Integer8<T>)
               return Unsupported {};
            else if constexpr (CT::Integer16<T>)
               return R {simde_mm512_mullo_epi16(lhs, rhs)};
            else if constexpr (CT::Integer32<T>)
               return R {simde_mm512_mullo_epi32(lhs, rhs)};
            else if constexpr (CT::Integer64<T>)
               return Unsupported{};
            else if constexpr (CT::Float<T>)
               return R {simde_mm512_mul_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)
               return R {simde_mm512_mul_pd(lhs, rhs)};
            else
               static_assert(false, "Unsupported type for 64-byte package");
         }
         else
      #endif
         static_assert(CT::False<T>, "Unsupported type");
      }
      
      /// Fallback multiplication                                             
      ///   @tparam SATURATE - whether to clamp to max if overflow occurs     
      template<bool SATURATE, class E> LANGULUS(INLINED)
      constexpr E MultiplyFallback(const E& lhs, const E& rhs) noexcept {
         if constexpr (SATURATE) {
            using WIDER = WiderSigned<E>;

            if constexpr (sizeof(WIDER) == sizeof(E) and CT::Integer<E>) {
               // If WIDER type isn't wider, perform the saturation     
               // by hand                                               
               if (rhs == 0)
                  return 0;

               constexpr E low = ::std::numeric_limits<E>::min();
               constexpr E hi  = ::std::numeric_limits<E>::max();
               if constexpr (CT::Signed<E>) {
                  if (rhs > 0)
                     return (lhs > hi / rhs) ? hi : ((lhs < low / rhs) ? low : lhs * rhs);
                  else
                     return (lhs > hi / (-rhs)) ? hi : ((lhs < low / (-rhs)) ? low : lhs * rhs);
               }
               else return (lhs > hi / rhs) ? hi : ((lhs < low / rhs) ? low : lhs * rhs);
            }
            else if constexpr (CT::Integer<E>)
               return Saturate<E>(static_cast<WIDER>(lhs) * static_cast<WIDER>(rhs));
            else
               return Saturate<E>(lhs * rhs);
         }
         else return lhs * rhs;
      }

      /// Get product of values as constexpr, if possible                     
      ///   @tparam SATURATE - whether to clamp to max if overflow occurs     
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the product scalar/vector                                 
      template<bool SATURATE = false, CT::NoIntent FORCE_OUT = void> LANGULUS(INLINED)
      constexpr auto MultiplyConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> E {
               return MultiplyFallback<SATURATE>(l, r);
            }
         );
      }
   
      /// Get product values as a register, if possible                       
      ///   @tparam SATURATE - whether to clamp to max if overflow occurs     
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the product scalar/vector/register                        
      template<bool SATURATE = false, CT::NoIntent FORCE_OUT = void> LANGULUS(INLINED)
      constexpr auto Multiply(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Multiplying (SIMD) as ", NameOf<R>());
               return MultiplySIMD<SATURATE>(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> E {
               LANGULUS_SIMD_VERBOSE("Multiplying (Fallback) ", l, " * ", r, " (", NameOf<E>(), ")");
               return MultiplyFallback<SATURATE>(l, r);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_WITH_SATURATION_API(Multiply)

} // namespace Langulus::SIMD
