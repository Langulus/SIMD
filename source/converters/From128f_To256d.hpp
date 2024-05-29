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

   /// Convert __m128 to __m256d                                              
   ///   @param v - the input __m128 register                                 
   ///   @return the resulting __m256d register                               
   LANGULUS(INLINED)
   simde__m256d ConvertFrom128f_To256d(const simde__m128& v) noexcept {
      //                                                                
      // Converting TO double[4]                                        
      //                                                                
      LANGULUS_SIMD_VERBOSE("Converting 256bit register from float[4] to double[4]");
      return simde_mm256_cvtps_pd(v);
   }

} // namespace Langulus::SIMD