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

   /// Convert __m256 to __m256d                                              
   ///   @param v - the input __m256 register                                 
   ///   @return the resulting __m256d register                               
   LANGULUS(INLINED)
   simde__m256d ConvertFrom256f_To256d(const simde__m256& v) noexcept {
      // float[4] -> double[4]                                          
      return simde_mm256_cvtps_pd(simde_mm256_castps256_ps128(v));
   }

} // namespace Langulus::SIMD