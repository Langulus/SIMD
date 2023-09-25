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

   /// Convert __m128d to any other register                                  
   ///   @tparam TO - the desired element type                                
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::SIMD REGISTER>
   LANGULUS(INLINED)
   auto ConvertFrom128d(const simde__m128d& v) noexcept {
      //                                                                
      // Converting FROM double[2]                                      
      //                                                                
      if constexpr (CT::SIMD128f<REGISTER>) {
         //                                                             
         // Converting TO float[2]                                      
         //                                                             
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to float[2]");
         return simde_mm_cvtpd_ps(v);
      }
      else if constexpr (CT::SIMD128i<REGISTER>) {
         //                                                             
         // Converting TO i8[2],  u8[2],  i16[2], u16[2]                
         //               i32[2], u32[2], i64[2], u64[2]                
         //                                                             
         if constexpr (CT::SignedInteger8<TO>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to i8[2]");
            auto
            vi32_16_8 = simde_mm_cvtpd_epi32(v);
            vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::UnsignedInteger8<TO>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to u8[2]");
            auto
            vi32_16_8 = simde_mm_cvtpd_epi32(v);
            vi32_16_8 = simde_mm_packus_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packus_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::SignedInteger16<TO>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to i16[2]");
            auto
            vi32_16 = simde_mm_cvtpd_epi32(v);
            vi32_16 = simde_mm_packs_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::UnsignedInteger16<TO>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to u16[2]");
            auto
            vi32_16 = simde_mm_cvtpd_epi32(v);
            vi32_16 = simde_mm_packus_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::Integer32<TO>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to u/i32[2]");
            return simde_mm_cvtpd_pi32(v);
         }
         else if constexpr (CT::SignedInteger64<TO>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to i64[2]");
            return _mm_cvtpd_epi64(v);
         }
         else if constexpr (CT::UnsignedInteger64<TO>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to u64[2]");
            return _mm_cvtpd_epu64(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128d to __m128i");
      }
      else LANGULUS_ERROR("Can't convert from __m128d to unsupported");
   }

} // namespace Langulus::SIMD