///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "From256f_To128f.hpp"
#include "From256f_To128d.hpp"
#include "From256f_To128i.hpp"
#include "From256f_To256d.hpp"
#include "From256f_To256i.hpp"
#include "From256f_To512d.hpp"
#include "From256f_To512i.hpp"


namespace Langulus::SIMD::Inner
{

   /// Convert __m256 to any other register                                   
   ///   @tparam TO - the desired element type                                
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::SIMD REGISTER>
   LANGULUS(INLINED)
   auto ConvertFrom256f(const simde__m256& v) noexcept {
      //                                                                
      // Converting FROM float[8]                                       
      //                                                                
      if constexpr (CT::SIMD128d<REGISTER>) {
         // float[2] -> double[2]                                       
         return simde_mm_cvtps_pd(simde_mm256_castps256_ps128(v));
      }
      else if constexpr (CT::SIMD128i<REGISTER>) {
         return ConvertFrom256f_To128i<TO>(v);
      }
      else if constexpr (CT::SIMD256i<REGISTER>) {
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
      else if constexpr (CT::SIMD256d<REGISTER>) {
         // float[4] -> double[4]                                       
         return simde_mm256_cvtps_pd(simde_mm256_castps256_ps128(v));
      }
      else LANGULUS_ERROR("Can't convert from __m256 to unsupported");
   }

} // namespace Langulus::SIMD