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

   /// Convert __m256 to __m128                                               
   ///   @param v - the input __m256 register                                 
   ///   @return the resulting __m128 register                                
   LANGULUS(INLINED)
   simde__m128 ConvertFrom256f_To128f(const simde__m256& v) noexcept {
      // float[8] -> float[4]                                           
      return simde_mm256_castps256_ps128(v);
   }

} // namespace Langulus::SIMD