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

   /// Convert __m256 to __m128i                                              
   ///   @tparam TO - the desired element type in __m128i                     
   ///   @param v - the input __m256 register                                 
   ///   @return the resulting __m128i register                               
   template<CT::Decayed TO> LANGULUS(INLINED)
   simde__m128i ConvertFrom256f_To128i(const simde__m256& v) noexcept {
      //                                                                
      // Converting TO  i8 [8], u8 [8], i16[8], u16[8]                  
      //                i32[4], u32[4], i64[2], u64[2]                  
      //                                                                
      if constexpr (CT::SignedInteger8<TO>) {
         // float[8] -> i8[8]                                           
         auto
         vi32_16_8 = simde_mm256_cvtps_epi32(v);
         vi32_16_8 = simde_mm256_packs_epi32(vi32_16_8, vi32_16_8);
         vi32_16_8 = simde_mm256_packs_epi16(vi32_16_8, vi32_16_8);
         return simde_mm256_castsi256_si128(vi32_16_8);
      }
      else if constexpr (CT::UnsignedInteger8<TO>) {
         // float[8] -> u8[8]                                           
         auto
         vu32_16_8 = simde_mm256_cvtps_epi32(v);
         vu32_16_8 = simde_mm256_packus_epi32(vu32_16_8, vu32_16_8);
         vu32_16_8 = simde_mm256_packus_epi16(vu32_16_8, vu32_16_8);
         return simde_mm256_castsi256_si128(vu32_16_8);
      }
      else if constexpr (CT::SignedInteger16<TO>) {
         // float[8] -> i16[8]                                          
         auto
         vi32_16 = simde_mm256_cvtps_epi32(v);
         vi32_16 = simde_mm256_packs_epi32(vi32_16, vi32_16);
         return simde_mm256_castsi256_si128(vi32_16);
      }
      else if constexpr (CT::UnsignedInteger16<TO>) {
         // float[8] -> u16[8]                                          
         auto
         vu32_16 = simde_mm256_cvtps_epi32(v);
         vu32_16 = simde_mm256_packus_epi32(vu32_16, vu32_16);
         return simde_mm256_castsi256_si128(vu32_16);
      }
      else if constexpr (CT::SignedInteger32<TO>) {
         // float[4] -> i32[4]                                          
         return simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));   
      }
      else if constexpr (CT::UnsignedInteger32<TO>) {
         // float[4] -> u32[4]                                          
         return simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));   
      }
      else if constexpr (CT::SignedInteger64<TO>) {
         // float[2] -> i64[2]                                          
         const auto vi32 = simde_mm256_cvtps_epi32(v);
         const auto vi32_128 = simde_mm256_castsi256_si128(vi32);
         const auto vi64 = simde_mm256_cvtepi32_epi64(vi32_128);
         return simde_mm256_castsi256_si128(vi64);
      }
      else if constexpr (CT::UnsignedInteger64<TO>) {
         // float[2] -> u64[2]                                          
         const auto vu32 = simde_mm256_cvtps_epi32(v);
         const auto vu32_128 = simde_mm256_castsi256_si128(vu32);
         const auto vu64 = simde_mm256_cvtepi32_epi64(vu32_128);
         return simde_mm256_castsi256_si128(vu64);
      }
      else LANGULUS_ERROR("Can't convert from __m256 to __m128i");
   }

} // namespace Langulus::SIMD