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

   /// Convert __m128 to __m128d                                              
   ///   @param v - the input __m128 register                                 
   ///   @return the resulting __m128d register                               
   LANGULUS(INLINED)
   simde__m128d ConvertFrom128f_To128d(const simde__m128& v) noexcept {
      //                                                                
      // Converting TO double[2]                                        
      //                                                                
      LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[2] to double[2]");
      return simde_mm_cvtps_pd(v);
   }

} // namespace Langulus::SIMD