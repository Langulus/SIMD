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

   /// Convert __m256d to any other register                                  
   ///   @tparam TO - the desired element type                                
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::SIMD REGISTER> LANGULUS(INLINED)
   auto ConvertFrom256d(const simde__m256d& v) noexcept {
      //                                                                
      // Converting FROM double[4]                                      
      //                                                                
      if constexpr (CT::SIMD128f<REGISTER>) {
         // double[4] -> float[4]                                       
         return simde_mm256_cvtpd_ps(v);
      }
      else if constexpr (CT::SIMD128i<REGISTER>) {
         //                                                             
         // Converting TO i8[4],  u8[4],  i16[4], u16[4]                
         //               i32[4], u32[4], i64[2], u64[2]                
         //                                                             
         if constexpr (CT::SignedInteger8<TO>) {
            // double[4] -> i8[4]                                       
            auto
            vi32_16_8 = simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
            vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, vi32_16_8);
            vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, vi32_16_8);
            return vi32_16_8;
         }
         else if constexpr (CT::UnsignedInteger8<TO>) {
            // double[4] -> u8[4]                                       
            auto
            vu32_16_8 = simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
            vu32_16_8 = simde_mm_packus_epi32(vu32_16_8, vu32_16_8);
            vu32_16_8 = simde_mm_packus_epi16(vu32_16_8, vu32_16_8);
            return vu32_16_8;
         }
         else if constexpr (CT::SignedInteger16<TO>) {
            // double[4] -> i16[4]                                      
            auto
            vi32_16 = simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
            vi32_16 = simde_mm_packs_epi32(vi32_16, vi32_16);
            return vi32_16;
         }
         else if constexpr (CT::UnsignedInteger16<TO>) {
            // double[4] -> u16[4]                                      
            auto
            vu32_16 = simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
            vu32_16 = simde_mm_packus_epi32(vu32_16, vu32_16);
            return vu32_16;
         }
         else if constexpr (CT::Integer32<TO>) {
            // double[4] -> i32[4]                                      
            // double[4] -> u32[4]                                      
            return simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
         }
         else if constexpr (CT::Integer64<TO>) {
            // double[2] -> i64[2]                                      
            // double[2] -> u64[2]                                      
            const auto v32 = simde_mm256_cvtpd_epi32(v);
            return simde_mm_unpacklo_epi32(v32, simde_mm_setzero_si128());
         }
         else LANGULUS_ERROR("Can't convert from __m256d to __m128i");
      }
      else LANGULUS_ERROR("Can't convert from __m256d to unsupported");
   }

} // namespace Langulus::SIMD