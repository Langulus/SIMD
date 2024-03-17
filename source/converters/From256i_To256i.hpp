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

   /// Convert __m256i to __m256i register with different layout              
   ///   @tparam TO - the desired __m256i can contain various kinds of ints   
   ///   @tparam FROM - the __m256i register can contain various kinds of ints
   ///   @param v - the input __m256i register                                
   ///   @return the resulting __m256i register                               
   template<CT::Decayed TO, CT::Decayed FROM> LANGULUS(INLINED)
   simde__m256i ConvertFrom256i_To256i(const simde__m256i& v) noexcept {
      LANGULUS_SIMD_VERBOSE("ConvertFrom256i_To256i");

      if constexpr (CT::Integer8<FROM>) {
         //                                                             
         // Converting TO i8[32], u8[32], i16[16], u16[16]              
         //               i32[8], u32[8], i64[4],  u64[4]               
         //                                                             
         if constexpr (CT::Integer8<TO>) {
            // u/i8[32] -> u/i8[32]                                     
            return v;
         }
         else if constexpr (CT::Integer16<TO>) {
            // u/i8[16] -> i/i16[16]                                    
            return simde_mm256_unpacklo_epi8(
               v, simde_mm256_setzero_si256()
            );
         }
         else if constexpr (CT::Integer32<TO>) {
            // u/i8[8] -> u/i32[8]                                      
            auto
            v16_32 = simde_mm256_unpacklo_epi8(
               v, simde_mm256_setzero_si256()
            );
            v16_32 = simde_mm256_unpacklo_epi16(
               v16_32, simde_mm256_setzero_si256()
            );
            return v16_32;
         }
         else if constexpr (CT::Integer64<TO>) {
            // u/i8[4] -> u/i64[4]                                      
            auto
            v16_32 = simde_mm256_unpacklo_epi8(
               v, simde_mm256_setzero_si256()
            );
            v16_32 = simde_mm256_unpacklo_epi16(
               v16_32, simde_mm256_setzero_si256()
            );
            return simde_mm256_cvtepi32_epi64(
               simde_mm256_castsi256_si128(v16_32));
         }
         else LANGULUS_ERROR("Can't convert from u/i8[32] to unsupported");
      }
      else if constexpr (CT::Integer16<FROM>) {
         //                                                             
         // Converting TO i8[16], u8[16], i16[16], u16[16]              
         //               i32[8], u32[8], i64[4],  u64[4]               
         //                                                             
         if constexpr (CT::SignedInteger8<TO>) {
            // u/i16[16] -> i8[16]                                      
            return simde_mm256_packs_epi16(v, v);
         }
         else if constexpr (CT::UnsignedInteger8<TO>) {
            // u/i16[16] -> u8[16]                                      
            return simde_mm256_packus_epi16(v, v);
         }
         else if constexpr (CT::Integer16<TO>) {
            // u/i16[16] -> u/i16[16]                                   
            return v;
         }
         else if constexpr (CT::Integer32<TO>) {
            // u/i16[8] -> u/i32[8]                                     
            return simde_mm256_unpacklo_epi16(
               v, simde_mm256_setzero_si256()
            );
         }
         else if constexpr (CT::Integer64<TO>) {
            // u/i16[4] -> u/i64[4]                                     
            auto
            v32_64 = simde_mm256_unpacklo_epi16(
               v, simde_mm256_setzero_si256()
            );
            v32_64 = simde_mm256_unpacklo_epi32(
               v32_64, simde_mm256_setzero_si256()
            );
            return v32_64;
         }
         else LANGULUS_ERROR("Can't convert from u/i16[16] to unsupported");
      }
      else if constexpr (CT::Integer32<FROM>) {
         //                                                             
         // Converting TO i8[8],  u8[8],  i16[8], u16[8]                
         //               i32[8], u32[8], i64[4], u64[4]                
         //                                                             
         if constexpr (CT::SignedInteger8<TO>) {
            // u/i32[8] -> i8[8]                                        
            auto
            v16_8 = simde_mm256_packs_epi32(v, v);
            v16_8 = simde_mm256_packs_epi16(v16_8, v16_8);
            return v16_8;
         }
         else if constexpr (CT::UnsignedInteger8<TO>) {
            // u/i32[8] -> u8[8]                                        
            auto
            v16_8 = simde_mm256_packus_epi32(v, v);
            v16_8 = simde_mm256_packus_epi16(v16_8, v16_8);
            return v16_8;
         }
         else if constexpr (CT::SignedInteger16<TO>) {
            // u/i32[8] -> i16[8]                                       
            return simde_mm256_packs_epi32(v, v);
         }
         else if constexpr (CT::UnsignedInteger16<TO>) {
            // u/i32[8] -> u16[8]                                       
            return simde_mm256_packus_epi32(v, v);
         }
         else if constexpr (CT::Integer32<TO>) {
            // u/i32[8] -> u/i32[8]                                     
            return v;
         }
         else if constexpr (CT::Integer64<TO>) {
            // u/i32[4] -> u/i64[4]                                     
            return simde_mm256_unpacklo_epi32(
               v, simde_mm256_setzero_si256()
            );
         }
         else LANGULUS_ERROR("Can't convert from u/i32[8] to unsupported");
      }
      else if constexpr (CT::Integer64<FROM>) {
         //                                                             
         // Converting TO i8[4],  u8[4],  i16[4], u16[4]                
         //               i32[4], u32[4], i64[4], u64[4]                
         //                                                             
         if constexpr (CT::Integer8<TO>) {
            // u/i64[4] -> u/i8[4]                                      
            auto
            v32_16_8 = lgls_pack_epi64(v, v);
            v32_16_8 = simde_mm256_packs_epi32(v32_16_8, v32_16_8);
            v32_16_8 = simde_mm256_packs_epi16(v32_16_8, v32_16_8);
            return v32_16_8;
         }
         else if constexpr (CT::SignedInteger16<TO>) {
            // u/i64[4] -> i16[4]                                       
            auto
            v32_16 = lgls_pack_epi64(v, v);
            v32_16 = simde_mm256_packs_epi32(v32_16, v32_16);
            return v32_16;
         }
         else if constexpr (CT::UnsignedInteger16<TO>) {
            // u/i64[4] -> u16[4]                                       
            auto
            v32_16 = lgls_pack_epi64(v, v);
            v32_16 = simde_mm256_packus_epi32(v32_16, v32_16);
            return v32_16;
         }
         else if constexpr (CT::Integer32<TO>) {
            // u/i64[4] -> u/i32[4]                                     
            return lgls_pack_epi64(v, v);
         }
         else if constexpr (CT::Integer64<TO>) {
            // u/i64[4] -> u/i64[4]                                     
            return v;
         }
         else LANGULUS_ERROR("Can't convert from u/i64[4] to unsupported");
      }
      else LANGULUS_ERROR("Can't convert unsupported FROM type");
   }

} // namespace Langulus::SIMD