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

   /// Convert __m128i to any other register                                  
   ///   @tparam TO - the desired element type                                
   ///   @tparam FROM - the previous element type, contained in REGISTER      
   ///                  (a 128i register can contain various kinds of ints)   
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::Decayed FROM, CT::SIMD REGISTER>
   LANGULUS(INLINED)
   auto ConvertFrom128i(const simde__m128i& v) noexcept {
      //                                                                
      // Converting FROM i/u8[16], i/u16[8], i/u32[4], i/u64[2]         
      //                                                                
      if constexpr (CT::SIMD128f<REGISTER>) {
         //                                                             
         // Converting TO float[4] or float[2]                          
         //                                                             
         if constexpr (CT::Integer8<FROM>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from u/i8[4] to float[4]");
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_ps(vi16_32);                                 
         }
         else if constexpr (CT::Integer16<FROM>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from u/i16[4] to float[4]");
            auto
            vi32 = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_ps(vi32);                                 
         }
         else if constexpr (CT::Integer32<FROM>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from u/i32[4] to float[4]");
            return simde_mm_cvtepi32_ps(v);
         }
         else if constexpr (CT::Integer64<FROM>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from u/i64[2] to float[2]");
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return simde_mm_cvtepi64_ps(v);
            #else
               return Unsupported {};
            #endif
         }
         else LANGULUS_ERROR("Can't convert from __m128i to __m128");
      }
      else if constexpr (CT::SIMD128d<REGISTER>) {
         //                                                             
         // Converting TO double[2]                                     
         //                                                             
         if constexpr (CT::Integer8<FROM>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from u/i8[2] to double[2]");
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_pd(vi16_32);                                 
         }
         else if constexpr (CT::Integer16<FROM>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from u/i16[2] to double[2]");
            auto
            vi32 = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_pd(vi32);                                 
         }
         else if constexpr (CT::Integer32<FROM>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from u/i32[2] to double[2]");
            return simde_mm_cvtepi32_pd(v);
         }
         else if constexpr (CT::Integer64<FROM>) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from u/i64[2] to double[2]");
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return simde_mm_cvtepi64_pd(v);
            #else
               return Unsupported {};
            #endif
         }
         else LANGULUS_ERROR("Can't convert from __m128i to __m128d");
      }
      else if constexpr (CT::SIMD128i<REGISTER>) {
         //                                                             
         // Converting TO i/u8[16], i/u16[8], i/u32[4], i/u64[2]        
         //                                                             
         if constexpr (CT::SignedInteger8<FROM>) {
            if constexpr (CT::Integer8<TO>) {
               return v;
            }
            else if constexpr (CT::Integer16<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from i8[8] to u/i16[8]");
               return simde_mm_cvtepi8_epi16(v);
            }
            else if constexpr (CT::Integer32<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from i8[4] to u/i32[4]");
               return simde_mm_cvtepi8_epi32(v);
            }
            else if constexpr (CT::SignedInteger64<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from i8[2] to i64[2]");
               return simde_mm_cvtepi8_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from i8 to unsupported");
         }
         else if constexpr (CT::UnsignedInteger8<FROM>) {
            if constexpr (CT::Integer8<TO>) {
               return v;
            }
            else if constexpr (CT::Integer16<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u8[8] to u/i16[8]");
               return simde_mm_cvtepu8_epi16(v);
            }
            else if constexpr (CT::Integer32<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u8[4] to u/i32[4]");
               return simde_mm_cvtepu8_epi32(v);
            }
            else if constexpr (CT::SignedInteger64<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u8[2] to i64[2]");
               return simde_mm_cvtepu8_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from u8 to unsupported");
         }
         else if constexpr (CT::SignedInteger16<FROM>) {
            if constexpr (CT::Integer8<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from i16[8] to u/i8[8]");
               return lgls_pack_epi16(v, v);
            }
            else if constexpr (CT::Integer16<TO>) {
               return v;
            }
            else if constexpr (CT::Integer32<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from i16[4] to u/i32[4]");
               return simde_mm_cvtepi16_epi32(v);
            }
            else if constexpr (CT::Integer64<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from i16[2] to u/i64[2]");
               return simde_mm_cvtepi16_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from i16 to unsupported");
         }
         else if constexpr (CT::UnsignedInteger16<FROM>) {
            if constexpr (CT::Integer8<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u16[8] to u/i8[8]");
               return lgls_pack_epi16(v, v);
            }
            else if constexpr (CT::Integer16<TO>) {
               return v;
            }
            else if constexpr (CT::Integer32<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u16[4] to u/i32[4]");
               return simde_mm_cvtepu16_epi32(v);
            }
            else if constexpr (CT::Integer64<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u16[2] to u/i64[2]");
               return simde_mm_cvtepu16_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from u16 to unsupported");
         }
         else if constexpr (CT::SignedInteger32<FROM>) {
            if constexpr (CT::Integer8<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from i32[4] to u/i8[4]");
               const auto t = lgls_pack_epi32(v, v);
               return lgls_pack_epi16(t, t);
            }
            else if constexpr (CT::Integer16<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from i32[4] to u/i16[4]");
               return lgls_pack_epi32(v, v);
            }
            else if constexpr (CT::Integer32<TO>) {
               return v;
            }
            else if constexpr (CT::SignedInteger64<TO>) {
               LANGULUS_ERROR("Can't convert from i32[2] to i64[2]");
            }
            else if constexpr (CT::UnsignedInteger64<TO>) {
               LANGULUS_ERROR("Can't convert from i32[2] to u64[2]");
            }
            else LANGULUS_ERROR("Can't convert from i32 to unsupported");
         }
         else if constexpr (CT::UnsignedInteger32<FROM>) {
            if constexpr (CT::SignedInteger8<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u32[4] to i8[4]");
               return simde_mm_packus_epi16(
                  simde_mm_packus_epi32(v, simde_mm_setzero_si128()),
                  simde_mm_setzero_si128()
               );
            }
            else if constexpr (CT::UnsignedInteger8<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u32[4] to u8[4]");
               return simde_mm_packus_epi16(
                  simde_mm_packus_epi32(v, simde_mm_setzero_si128()),
                  simde_mm_setzero_si128()
               );
            }
            else if constexpr (CT::SignedInteger16<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u32[4] to i16[4]");
               return simde_mm_packus_epi32(v, simde_mm_setzero_si128());
            }
            else if constexpr (CT::UnsignedInteger16<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u32[4] to u16[4]");
               return simde_mm_packus_epi32(v, simde_mm_setzero_si128());
            }
            else if constexpr (CT::Integer32<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u/i32[4] to u/i32[4]");
               return v;
            }
            else if constexpr (CT::SignedInteger64<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u32[2] to i64[2]");
               return simde_mm_cvtepu32_epi64(v);
            }
            else if constexpr (CT::UnsignedInteger64<TO>) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from u32[2] to u64[2]");
               return simde_mm_cvtepu32_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from u32 to unsupported");
         }
         else if constexpr (CT::SignedInteger64<FROM>) {
            if constexpr (CT::SignedInteger8<TO>) {
               // i64[2] -> i8[2]                                       
               LANGULUS_ERROR("Can't convert from i64[2] to i8[2]");
            }
            else if constexpr (CT::UnsignedInteger8<TO>) {
               // i64[2] -> u8[2]                                       
               LANGULUS_ERROR("Can't convert from i64[2] to u8[2]");
            }
            else if constexpr (CT::SignedInteger16<TO>) {
               // i64[2] -> i16[2]                                      
               LANGULUS_ERROR("Can't convert from i64[2] to i16[2]");
            }
            else if constexpr (CT::UnsignedInteger16<TO>) {
               // i64[2] -> u16[2]                                      
               LANGULUS_ERROR("Can't convert from i64[2] to u16[2]");
            }
            else if constexpr (CT::SignedInteger32<TO>) {
               // i64[2] -> i32[2]                                      
               LANGULUS_ERROR("Can't convert from i64[2] to i32[2]");
            }
            else if constexpr (CT::UnsignedInteger32<TO>) {
               // i64[2] -> u32[2]                                      
               LANGULUS_ERROR("Can't convert from i64[2] to u32[2]");
            }
            else if constexpr (CT::SignedInteger64<TO>) {
               // i64[2] -> i64[2]                                      
               LANGULUS_ERROR("Can't convert from i64[2] to i64[2]");
            }
            else if constexpr (CT::UnsignedInteger64<TO>) {
               // i64[2] -> u64[2]                                      
               LANGULUS_ERROR("Can't convert from i64[2] to u64[2]");
            }
            else LANGULUS_ERROR("Can't convert from i64 to unsupported");
         }
         else if constexpr (CT::UnsignedInteger64<FROM>) {
            if constexpr (CT::SignedInteger8<TO>) {
               // u64[2] -> i8[2]                                       
               LANGULUS_ERROR("Can't convert from u64[2] to i8[2]");
            }
            else if constexpr (CT::UnsignedInteger8<TO>) {
               // u64[2] -> u8[2]                                       
               LANGULUS_ERROR("Can't convert from u64[2] to u8[2]");
            }
            else if constexpr (CT::SignedInteger16<TO>) {
               // u64[2] -> i16[2]                                      
               LANGULUS_ERROR("Can't convert from u64[2] to i16[2]");
            }
            else if constexpr (CT::UnsignedInteger16<TO>) {
               // u64[2] -> u16[2]                                      
               LANGULUS_ERROR("Can't convert from u64[2] to u16[2]");
            }
            else if constexpr (CT::SignedInteger32<TO>) {
               // u64[2] -> i32[2]                                      
               LANGULUS_ERROR("Can't convert from u64[2] to i32[2]");
            }
            else if constexpr (CT::UnsignedInteger32<TO>) {
               // u64[2] -> u32[2]                                      
               LANGULUS_ERROR("Can't convert from u64[2] to u32[2]");
            }
            else if constexpr (CT::SignedInteger64<TO>) {
               // u64[2] -> i64[2]                                      
               LANGULUS_ERROR("Can't convert from u64[2] to i64[2]");
            }
            else if constexpr (CT::UnsignedInteger64<TO>) {
               // u64[2] -> u64[2]                                      
               LANGULUS_ERROR("Can't convert from u64[2] to u64[2]");
            }
            else LANGULUS_ERROR("Can't convert from u64 to unsupported");
         }
         else LANGULUS_ERROR("Can't convert from __m128i to __m128i");
      }
      else if constexpr (CT::SIMD128d<REGISTER>) {
         //                                                             
         // Converting TO double[2]                                     
         //                                                             
         LANGULUS_ERROR("Can't convert from __m128i to __m128d");
      }
      else
#if LANGULUS_SIMD(256BIT)

#endif
      LANGULUS_ERROR("Can't convert from __m128i to unsupported");
   }

} // namespace Langulus::SIMD