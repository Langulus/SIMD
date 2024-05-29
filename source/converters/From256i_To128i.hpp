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

   /// Convert __m256i to __m128i register                                    
   ///   @tparam TO - the desired __m128i can contain various kinds of ints   
   ///   @tparam FROM - the 256i register can contain various kinds of ints   
   ///   @param v - the input __m256i register                                
   ///   @return the resulting __m128i register                               
   template<CT::Decayed TO, CT::Decayed FROM> LANGULUS(INLINED)
   simde__m128i ConvertFrom256i_To128i(const simde__m256i& v) noexcept {
      LANGULUS_SIMD_VERBOSE("ConvertFrom256i_To128i");

      if constexpr (CT::SignedInteger8<FROM>) {
         //                                                             
         // Converting TO i8[16], u8[16], i16[8], u16[8]                
         //               i32[4], u32[4], i64[2], u64[2]                
         //                                                             
         if constexpr (CT::Integer8<TO>) {
            // i8[16] -> u/i8[16]                                       
            return simde_mm256_castsi256_si128(v);
         }
         else if constexpr (CT::Integer16<TO>) {
            // i8[8] -> i/i16[8]                                        
            return simde_mm_unpacklo_epi8(
               simde_mm256_castsi256_si128(v),
               simde_mm_setzero_si128()
            );
         }
         else if constexpr (CT::Integer32<TO>) {
            // i8[4] -> u/i32[4]                                        
            auto
            v16_32 = simde_mm_unpacklo_epi8(
               simde_mm256_castsi256_si128(v),
               simde_mm_setzero_si128()
            );
            v16_32 = simde_mm_unpacklo_epi16(
               v16_32,
               simde_mm_setzero_si128()
            );
            return v16_32;
         }
         else if constexpr (CT::Integer64<TO>) {
            // i8[2] -> u/i64[2]                                        
            auto
            v16_32 = simde_mm_unpacklo_epi8(
               simde_mm256_castsi256_si128(v),
               simde_mm_setzero_si128()
            );
            v16_32 = simde_mm_unpacklo_epi16(
               v16_32,
               simde_mm_setzero_si128()
            );
            return simde_mm_cvtepi32_epi64(v16_32);
         }
         else LANGULUS_ERROR("Can't convert from i8 to unsupported");
      }
      else if constexpr (CT::UnsignedInteger8<FROM>) {
         //                                                             
         // Converting TO i8[16], u8[16], i16[8], u16[8]                
         //               i32[4], u32[4], i64[2], u64[2]                
         //                                                             
         if constexpr (CT::Integer8<TO>) {
            // u8[16] -> u/i8[16]                                       
            return simde_mm256_castsi256_si128(v);
         }
         else if constexpr (CT::Integer16<TO>) {
            // u8[8] -> u/i16[8]                                        
            return simde_mm_unpacklo_epi8(
               simde_mm256_castsi256_si128(v),
               simde_mm_setzero_si128()
            );
         }
         else if constexpr (CT::Integer32<TO>) {
            // u8[4] -> u/i32[4]                                        
            auto
            v16_32 = simde_mm_unpacklo_epi8(
               simde_mm256_castsi256_si128(v),
               simde_mm_setzero_si128()
            );
            v16_32 = simde_mm_unpacklo_epi16(
               v16_32,
               _mm_setzero_si128()
            );
            return v16_32;
         }
         else if constexpr (CT::Integer64<TO>) {
            // u8[2] -> u/i64[2]                                        
            auto
            v16_32 = simde_mm_unpacklo_epi8(
               simde_mm256_castsi256_si128(v),
               simde_mm_setzero_si128()
            );
            v16_32 = simde_mm_unpacklo_epi16(
               v16_32,
               simde_mm_setzero_si128()
            );
            return simde_mm_cvtepi32_epi64(v16_32);
         }
         else LANGULUS_ERROR("Can't convert from u8 to unsupported");
      }
      else if constexpr (CT::SignedInteger16<FROM>) {
         //                                                             
         // Converting TO i8[8],  u8[8],  i16[8], u16[8]                
         //               i32[4], u32[4], i64[2], u64[2]                
         //                                                             
         if constexpr (CT::SignedInteger8<TO>) {
            // i16[8] -> i8[8]                                          
            LANGULUS_ERROR("Can't convert from i16[8] to i8[8]");
         }
         else if constexpr (CT::UnsignedInteger8<TO>) {
            // i16[8] -> u8[8]                                          
            LANGULUS_ERROR("Can't convert from i16[8] to u8[8]");
         }
         else if constexpr (CT::UnsignedInteger16<TO>) {
            // i16[8] -> u16[8]                                         
            LANGULUS_ERROR("Can't convert from i16[8] to u16[8]");
         }
         else if constexpr (CT::SignedInteger32<TO>) {
            // i16[4] -> i32[4]                                         
            LANGULUS_ERROR("Can't convert from i16[4] to i32[4]");
         }
         else if constexpr (CT::UnsignedInteger32<TO>) {
            // i16[4] -> u32[4]                                         
            LANGULUS_ERROR("Can't convert from i16[4] to u32[4]");
         }
         else if constexpr (CT::SignedInteger64<TO>) {
            // i16[2] -> i64[2]                                         
            LANGULUS_ERROR("Can't convert from i16[2] to i64[2]");
         }
         else if constexpr (CT::UnsignedInteger64<TO>) {
            // i16[2] -> u64[2]                                         
            LANGULUS_ERROR("Can't convert from i16[2] to u64[2]");
         }
         else LANGULUS_ERROR("Can't convert from i16 to unsupported");
      }
      else if constexpr (CT::UnsignedInteger16<FROM>) {
         //                                                             
         // Converting TO i8[8],  u8[8],  i16[8], u16[8]                
         //               i32[4], u32[4], i64[2], u64[2]                
         //                                                             
         if constexpr (CT::SignedInteger8<TO>) {
            // u16[8] -> i8[8]                                          
            LANGULUS_ERROR("Can't convert from u16[8] to i8[8]");
         }
         else if constexpr (CT::UnsignedInteger8<TO>) {
            // u16[8] -> u8[8]                                          
            LANGULUS_ERROR("Can't convert from u16[8] to u8[8]");
         }
         else if constexpr (CT::SignedInteger16<TO>) {
            // u16[8] -> i16[8]                                         
            LANGULUS_ERROR("Can't convert from u16[8] to i16[8]");
         }
         else if constexpr (CT::SignedInteger32<TO>) {
            // u16[4] -> i32[4]                                         
            LANGULUS_ERROR("Can't convert from u16[4] to i32[4]");
         }
         else if constexpr (CT::UnsignedInteger32<TO>) {
            // u16[4] -> u32[4]                                         
            LANGULUS_ERROR("Can't convert from u16[4] to u32[4]");
         }
         else if constexpr (CT::SignedInteger64<TO>) {
            // u16[2] -> i64[2]                                         
            LANGULUS_ERROR("Can't convert from u16[2] to i64[2]");
         }
         else if constexpr (CT::UnsignedInteger64<TO>) {
            // u16[2] -> u64[2]                                         
            LANGULUS_ERROR("Can't convert from u16[2] to u64[2]");
         }
         else LANGULUS_ERROR("Can't convert from u16 to unsupported");
      }
      else if constexpr (CT::Integer32<FROM>) {
         //                                                             
         // Converting TO i8[4],  u8[4],  i16[4], u16[4]                
         //               i32[4], u32[4], i64[2], u64[2]                
         //                                                             
         if constexpr (CT::SignedInteger8<TO>) {
            // u/i32[4] -> i8[4]                                        
            auto
            v16_8 = simde_mm256_castsi256_si128(v);
            v16_8 = simde_mm_packs_epi32(v16_8, v16_8);
            v16_8 = simde_mm_packs_epi16(v16_8, v16_8);
            return v16_8;
         }
         else if constexpr (CT::UnsignedInteger8<TO>) {
            // u/i32[4] -> u8[4]                                        
            auto
            v16_8 = simde_mm256_castsi256_si128(v);
            v16_8 = simde_mm_packus_epi32(v16_8, v16_8);
            v16_8 = simde_mm_packus_epi16(v16_8, v16_8);
            return v16_8;
         }
         else if constexpr (CT::SignedInteger16<TO>) {
            // u/i32[4] -> i16[4]                                       
            auto
            v16_8 = simde_mm256_castsi256_si128(v);
            v16_8 = simde_mm_packs_epi32(v16_8, v16_8);
            return v16_8;
         }
         else if constexpr (CT::UnsignedInteger16<TO>) {
            // u/i32[4] -> u16[4]                                       
            auto
            v16_8 = simde_mm256_castsi256_si128(v);
            v16_8 = simde_mm_packus_epi32(v16_8, v16_8);
            return v16_8;
         }
         else if constexpr (CT::Integer32<TO>) {
            // u/i32[4] -> u/i32[4]                                     
            return simde_mm256_castsi256_si128(v);
         }
         else if constexpr (CT::SignedInteger64<TO>) {
            // u/i32[2] -> i64[2]                                       
            const auto v64 = simde_mm256_cvtepi32_epi64(
               simde_mm256_castsi256_si128(v));
            return simde_mm256_castsi256_si128(v64);
         }
         else if constexpr (CT::UnsignedInteger64<TO>) {
            // u/i32[2] -> u64[2]                                       
            const auto v64 = simde_mm256_cvtepu32_epi64(
               simde_mm256_castsi256_si128(v));
            return simde_mm256_castsi256_si128(v64);
         }
         else LANGULUS_ERROR("Can't convert from i32 to unsupported");
      }
      else if constexpr (CT::SignedInteger64<FROM>) {
         //                                                             
         // Converting TO i8[2],  u8[2],  i16[2], u16[2]                
         //               i32[2], u32[2], i64[2], u64[2]                
         //                                                             
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
         else if constexpr (CT::UnsignedInteger64<TO>) {
            // i64[2] -> u64[2]                                         
            LANGULUS_ERROR("Can't convert from i64[2] to u64[2]");
         }
         else LANGULUS_ERROR("Can't convert from i64 to unsupported");
      }
      else if constexpr (CT::UnsignedInteger64<FROM>) {
         //                                                             
         // Converting TO i8[2],  u8[2],  i16[2], u16[2]                
         //               i32[2], u32[2], i64[2], u64[2]                
         //                                                             
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
         else LANGULUS_ERROR("Can't convert from u64 to unsupported");
      }
      else LANGULUS_ERROR("Can't convert from __m256i to __m128i");
   }

} // namespace Langulus::SIMD