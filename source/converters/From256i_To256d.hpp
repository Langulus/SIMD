///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "../Common.hpp"


namespace Langulus::SIMD::Inner
{

   /// Convert __m256i to __m256d                                             
   ///   @tparam FROM - the 256i register can contain various kinds of ints   
   ///   @param v - the __m256i register                                      
   ///   @return the resulting __m256d register                               
   template<CT::Decayed FROM> LANGULUS(INLINED)
   simde__m256d ConvertFrom256i_To256d(const simde__m256i& v) noexcept {
      LANGULUS_SIMD_VERBOSE("ConvertFrom256i_To256d");

      //                                                                
      // Converting TO double[4]                                        
      //                                                                
      if constexpr (CT::SignedInteger8<FROM>) {
         // i8[4] -> double[4]                                          
         const auto v32 = simde_mm256_cvtepi8_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm256_cvtepi32_pd(
            simde_mm256_castsi256_si128(v32));
      }
      else if constexpr (CT::UnsignedInteger8<FROM>) {
         // u8[4] -> double[4]                                          
         const auto v32 = simde_mm256_cvtepu8_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm256_cvtepi32_pd(
            simde_mm256_castsi256_si128(v32));
      }
      else if constexpr (CT::SignedInteger16<FROM>) {
         // i16[4] -> double[4]                                         
         const auto v32 = simde_mm256_cvtepi16_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm256_cvtepi32_pd(
            simde_mm256_castsi256_si128(v32));
      }
      else if constexpr (CT::UnsignedInteger16<FROM>) {
         // u16[4] -> double[4]                                         
         const auto v32 = simde_mm256_cvtepu16_epi32(
            simde_mm256_castsi256_si128(v));
         return simde_mm256_cvtepi32_pd(
            simde_mm256_castsi256_si128(v32));
      }
      else if constexpr (CT::Integer32<FROM>) {
         // i/u32[4] -> double[4]                                       
         return simde_mm256_cvtepi32_pd(simde_mm256_castsi256_si128(v));
      }
      else if constexpr (CT::SignedInteger64<FROM>) {
         // i64[2] -> double[2]                                         
         //TODO generalize this when 512 stuff is added to SIMDe        
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return simde_mm256_cvtepi64_pd(v);
         #else
            auto m1 = int64_to_double_full(simde_mm256_extracti128_si256(v, 0));
            auto m2 = int64_to_double_full(simde_mm256_extracti128_si256(v, 1));
            return simde_mm256_set_m128d(m1, m2);
         #endif
      }
      else if constexpr (CT::UnsignedInteger64<FROM>) {
         // u64[2] -> double[2]                                         
         //TODO generalize this when 512 stuff is added to SIMDe        
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return simde_mm_cvtepu64_pd(simde_mm256_extracti128_si256(v, 0));
         #else
            auto m1 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 0));
            auto m2 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 1));
            return simde_mm256_set_m128d(m1, m2);
         #endif
      }
      else LANGULUS_ERROR("Can't convert from __m256i to __m128d");
   }

} // namespace Langulus::SIMD