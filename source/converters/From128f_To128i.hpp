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

   /// Convert __m128 to __m128i register                                     
   ///   @tparam TO - the desired element type inside __m128i                 
   ///   @param v - the input __m128 register                                 
   ///   @return the resulting __m128i register                               
   template<CT::Decayed TO>
   LANGULUS(INLINED)
   simde__m128i ConvertFrom128f_To128i(const simde__m128& v) noexcept {
      //                                                                
      // Converting TO i8 [4], u8 [4], i16[4], u16[4]                   
      //               i32[4], u32[4], i64[2], u64[2]                   
      //                                                                
      if constexpr (CT::SignedInteger8<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to i8[4]");
         auto 
         vi32_16_8 = simde_mm_cvtps_epi32(v);
         vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, simde_mm_setzero_si128());
         vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, simde_mm_setzero_si128());
         return vi32_16_8;
      }
      else if constexpr (CT::UnsignedInteger8<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to u8[4]");
         auto
         vi32_16_8 = simde_mm_cvtps_epi32(v);
         vi32_16_8 = simde_mm_packus_epi32(vi32_16_8, simde_mm_setzero_si128());
         vi32_16_8 = simde_mm_packus_epi16(vi32_16_8, simde_mm_setzero_si128());
         return vi32_16_8;
      }
      else if constexpr (CT::SignedInteger16<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to i16[4]");
         auto
         vi32_16 = simde_mm_cvtps_epi32(v);
         vi32_16 = simde_mm_packs_epi32(vi32_16, simde_mm_setzero_si128());
         return vi32_16;
      }
      else if constexpr (CT::UnsignedInteger16<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to u16[4]");
         auto
         vi32_16 = simde_mm_cvtps_epi32(v);
         vi32_16 = simde_mm_packus_epi32(vi32_16, simde_mm_setzero_si128());
         return vi32_16;
      }
      else if constexpr (CT::SignedInteger32<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to i32[4]");
         return simde_mm_cvtps_epi32(v);
      }
      else if constexpr (CT::UnsignedInteger32<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to i32[4]");
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            return simde_mm_cvtps_epu32(v);
         #else
            return Unsupported {};
         #endif
      }
      else if constexpr (CT::SignedInteger64<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[2] to i64[2]");
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return simde_mm_cvtps_epi64(v);
         #else
            return Unsupported {};
         #endif
      }
      else if constexpr (CT::UnsignedInteger64<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[2] to i64[2]");
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return simde_mm_cvtps_epu64(v);
         #else
            return Unsupported {};
         #endif
      }
      else LANGULUS_ERROR("Can't convert from __m128 to __m128i");
   }

} // namespace Langulus::SIMD