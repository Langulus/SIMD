///                                                                           
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

   /// Convert __m128 to __m256i register                                     
   ///   @tparam TO - the desired element type inside __m256i                 
   ///   @param v - the input __m128 register                                 
   ///   @return the resulting __m256i register                               
   template<CT::Decayed TO> LANGULUS(INLINED)
   simde__m256i ConvertFrom128f_To256i(const simde__m128& v) noexcept {
      //                                                                
      // Converting TO i64[4], u64[4]                                   
      //                                                                
      if constexpr (CT::SignedInteger64<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 256bit register from float[4] to i64[4]");
         return _mm256_cvtps_epi64(v);
      }
      else if constexpr (CT::UnsignedInteger64<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 256bit register from float[4] to u64[4]");
         return _mm256_cvtps_epu64(v);
      }
      else LANGULUS_ERROR("Can't convert from __m128 to __m256i");
   }

} // namespace Langulus::SIMD