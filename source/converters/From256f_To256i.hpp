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

   /// Convert __m256 to __m256i                                              
   ///   @tparam TO - the desired element type in __m256i                     
   ///   @param v - the input __m256 register                                 
   ///   @return the resulting __m256i register                               
   template<CT::Decayed TO> LANGULUS(INLINED)
   simde__m256i ConvertFrom256f_To256i(const simde__m256& v) noexcept {
      //                                                                
      // Converting TO i64[4], u64[4]                                   
      //                                                                
      if constexpr (CT::SignedInteger64<TO>) {
         // float[4] -> i64[4]                                          
         const auto v32 = simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));
         return simde_mm256_cvtepi32_epi64(v32);
      }
      else if constexpr (CT::UnsignedInteger64<TO>) {
         // float[4] -> u64[4]                                          
         const auto v32 = simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));
         return simde_mm256_cvtepu32_epi64(v32);
      }
      else LANGULUS_ERROR("Can't convert from __m256 to __m256i");
   }

} // namespace Langulus::SIMD