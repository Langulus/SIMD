///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "../Common.hpp"


namespace Langulus::SIMD::Inner
{

   /// Convert __m256i to __m256                                              
   ///   @tparam FROM - the 256i register can contain various kinds of ints   
   ///   @param v - the __m256i register                                      
   ///   @return the resulting __m256 register                                
   template<CT::Decayed FROM> LANGULUS(INLINED)
   simde__m256 ConvertFrom256i_To256f(const simde__m256i& v) noexcept {
      LANGULUS_SIMD_VERBOSE("ConvertFrom256i_To256f");

      //                                                                
      // Converting TO float[8]                                         
      //                                                                
      if constexpr (CT::SignedInteger8<FROM>) {
         // i8[8] -> float[8]                                           
         const auto v32 = simde_mm256_cvtepi8_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm256_cvtepi32_ps(v32);
      }
      else if constexpr (CT::UnsignedInteger8<FROM>) {
         // u8[8] -> float[8]                                           
         const auto v32 = simde_mm256_cvtepu8_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm256_cvtepi32_ps(v32);
      }
      else if constexpr (CT::SignedInteger16<FROM>) {
         // i16[8] -> float[8]                                          
         const auto v32 = simde_mm256_cvtepi16_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm256_cvtepi32_ps(v32);
      }
      else if constexpr (CT::UnsignedInteger16<FROM>) {
         // u16[8] -> float[8]                                          
         const auto v32 = simde_mm256_cvtepu16_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm256_cvtepi32_ps(v32);
      }
      else if constexpr (CT::Integer32<FROM>) {
         // i/u32[8] -> float[8]                                        
         return simde_mm256_cvtepi32_ps(v);   
      }
      else if constexpr (CT::SignedInteger64<FROM>) {
         // i64[4] -> float[4]                                          
         //TODO generalize this when 512 stuff is added to SIMDe        
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return simde_mm256_cvtepi64_ps(v);
         #else
            auto m1 = int64_to_double_full(simde_mm256_extracti128_si256(v, 0));
            auto m2 = int64_to_double_full(simde_mm256_extracti128_si256(v, 1));
            return simde_mm256_set_m128(
               simde_mm_movelh_ps(simde_mm_cvtpd_ps(m1), simde_mm_cvtpd_ps(m2)),
               simde_mm_setzero_ps()
            );
         #endif
      }
      else if constexpr (CT::UnsignedInteger64<FROM>) {
         // u64[4] -> float[4]                                          
         //TODO generalize this when 512 stuff is added to SIMDe        
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return simde_mm256_cvtepu64_ps(v);
         #else
            auto m1 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 0));
            auto m2 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 1));
            return simde_mm256_set_m128(
               simde_mm_movelh_ps(simde_mm_cvtpd_ps(m1), simde_mm_cvtpd_ps(m2)),
               simde_mm_setzero_ps()
            );
         #endif
      }
      else LANGULUS_ERROR("Can't convert from __m256i to __m128");
   }

} // namespace Langulus::SIMD