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

   /// Convert __m256i to __m128d                                             
   ///   @tparam FROM - the 256i register can contain various kinds of ints   
   ///   @param v - the __m256i register                                      
   ///   @return the resulting __m128d register                               
   template<CT::Decayed FROM>
   LANGULUS(INLINED)
   simde__m128d ConvertFrom256i_To128d(const simde__m256i& v) noexcept {
      LANGULUS_SIMD_VERBOSE("ConvertFrom256i_To128d");

      //                                                                
      // Converting TO double[2]                                        
      //                                                                
      if constexpr (CT::SignedInteger8<FROM>) {
         // i8[2] -> double[2]                                          
         const auto v32 = simde_mm_cvtepi8_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm_cvtepi32_pd(v32);
      }
      else if constexpr (CT::UnsignedInteger8<FROM>) {
         // u8[2] -> double[2]                                          
         const auto v32 = simde_mm_cvtepu8_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm_cvtepi32_pd(v32);
      }
      else if constexpr (CT::SignedInteger16<FROM>) {
         // i16[2] -> double[2]                                         
         const auto v32 = simde_mm_cvtepi16_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm_cvtepi32_pd(v32);
      }
      else if constexpr (CT::UnsignedInteger16<FROM>) {
         // u16[2] -> double[2]                                         
         const auto v32 = simde_mm_cvtepu16_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm_cvtepi32_pd(v32);
      }
      else if constexpr (CT::Integer32<FROM>) {
         // i/u32[2] -> double[2]                                       
         return simde_mm_cvtepi32_pd(simde_mm256_castsi256_si128(v));
      }
      else if constexpr (CT::SignedInteger64<FROM>) {
         // i64[2] -> double[2]                                         
         //TODO generalize this when 512 stuff is added to SIMDe        
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return simde_mm_cvtepi64_pd(simde_mm256_extracti128_si256(v, 0));
         #else
            return int64_to_double_full(simde_mm256_extracti128_si256(v, 0));
         #endif
      }
      else if constexpr (CT::UnsignedInteger64<FROM>) {
         // u64[2] -> double[2]                                         
         //TODO generalize this when 512 stuff is added to SIMDe        
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return simde_mm_cvtepu64_pd(simde_mm256_extracti128_si256(v, 0));
         #else
            return uint64_to_double_full(simde_mm256_extracti128_si256(v, 0));
         #endif
      }
      else LANGULUS_ERROR("Can't convert from __m256i to __m128d");
   }

} // namespace Langulus::SIMD