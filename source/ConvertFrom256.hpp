///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Load.hpp"


namespace Langulus::SIMD::Inner
{

   /// Convert __m256 to any other register                                   
   ///   @tparam TO - the desired element type                                
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::SIMD REGISTER>
   LANGULUS(INLINED)
   auto ConvertFrom256(const simde__m256& v) noexcept {
      //                                                                
      // Converting FROM float[8]                                       
      //                                                                
      if constexpr (CT::SIMD128d<REGISTER>) {
         // float[2] -> double[2]                                       
         return simde_mm_cvtps_pd(simde_mm256_castps256_ps128(v));
      }
      else if constexpr (CT::SIMD128i<REGISTER>) {
         //                                                             
         // Converting TO  i8[8],  u8[8],  i16[8], u16[8]               
         //                i32[4], u32[4], i64[2], u64[2]               
         //                                                             
         if constexpr (CT::SignedInteger8<TO>) {
            // float[8] -> i8[8]                                        
            auto
            vi32_16_8 = simde_mm256_cvtps_epi32(v);
            vi32_16_8 = simde_mm256_packs_epi32(vi32_16_8, vi32_16_8);
            vi32_16_8 = simde_mm256_packs_epi16(vi32_16_8, vi32_16_8);
            return simde_mm256_castsi256_si128(vi32_16_8);
         }
         else if constexpr (CT::UnsignedInteger8<TO>) {
            // float[8] -> u8[8]                                        
            auto
            vu32_16_8 = simde_mm256_cvtps_epi32(v);
            vu32_16_8 = simde_mm256_packus_epi32(vu32_16_8, vu32_16_8);
            vu32_16_8 = simde_mm256_packus_epi16(vu32_16_8, vu32_16_8);
            return simde_mm256_castsi256_si128(vu32_16_8);
         }
         else if constexpr (CT::SignedInteger16<TO>) {
            // float[8] -> i16[8]                                       
            auto
            vi32_16 = simde_mm256_cvtps_epi32(v);
            vi32_16 = simde_mm256_packs_epi32(vi32_16, vi32_16);
            return simde_mm256_castsi256_si128(vi32_16);
         }
         else if constexpr (CT::UnsignedInteger16<TO>) {
            // float[8] -> u16[8]                                       
            auto
            vu32_16 = simde_mm256_cvtps_epi32(v);
            vu32_16 = simde_mm256_packus_epi32(vu32_16, vu32_16);
            return simde_mm256_castsi256_si128(vu32_16);
         }
         else if constexpr (CT::SignedInteger32<TO>) {
            // float[4] -> i32[4]                                       
            return simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));
         }
         else if constexpr (CT::UnsignedInteger32<TO>) {
            // float[4] -> u32[4]                                       
            return simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));
         }
         else if constexpr (CT::SignedInteger64<TO>) {
            // float[2] -> i64[2]                                       
            const auto vi32 = simde_mm256_cvtps_epi32(v);
            const auto vi32_128 = simde_mm256_castsi256_si128(vi32);
            const auto vi64 = simde_mm256_cvtepi32_epi64(vi32_128);
            return simde_mm256_castsi256_si128(vi64);
         }
         else if constexpr (CT::UnsignedInteger64<TO>) {
            // float[2] -> u64[2]                                       
            const auto vu32 = simde_mm256_cvtps_epi32(v);
            const auto vu32_128 = simde_mm256_castsi256_si128(vu32);
            const auto vu64 = simde_mm256_cvtepi32_epi64(vu32_128);
            return simde_mm256_castsi256_si128(vu64);
         }
         else LANGULUS_ERROR("Can't convert from __m256 to __m128i");
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

   /// Convert __m256d to any other register                                  
   ///   @tparam TO - the desired element type                                
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::SIMD REGISTER>
   LANGULUS(INLINED)
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

   /// Convert __m256i to any other register                                  
   ///   @tparam TO - the desired element type                                
   ///   @tparam FROM - the previous element type, contained in REGISTER      
   ///                  (a 256i register can contain various kinds of ints)   
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::Decayed FROM, CT::SIMD REGISTER>
   LANGULUS(INLINED)
   auto ConvertFrom256i(const simde__m256i& v) noexcept {
      //                                                                
      // Converting FROM i8[32], u8[32], i16[16], u16[16]               
      //                 i32[8], u32[8], i64[4],  u64[4]                
      //                                                                
      if constexpr (CT::SIMD128f<REGISTER>) {
         //                                                             
         // Converting TO float[4]                                      
         //                                                             
         if constexpr (CT::SignedInteger8<FROM>) {
            // i8[4] -> float[4]                                        
            const auto v32 = simde_mm_cvtepi8_epi32(
               simde_mm256_castsi256_si128(v));
            return simde_mm_cvtepi32_ps(v32);
         }
         else if constexpr (CT::UnsignedInteger8<FROM>) {
            // u8[4] -> float[4]                                        
            const auto v32 = simde_mm_cvtepu8_epi32(
               simde_mm256_castsi256_si128(v));
            return simde_mm_cvtepi32_ps(v32);
         }
         else if constexpr (CT::SignedInteger16<FROM>) {
            // i16[4] -> float[4]                                       
            const auto v32 = simde_mm_cvtepi16_epi32(
               simde_mm256_castsi256_si128(v));
            return simde_mm_cvtepi32_ps(v32);
         }
         else if constexpr (CT::UnsignedInteger16<FROM>) {
            // u16[4] -> float[4]                                       
            const auto v32 = simde_mm_cvtepu16_epi32(
               simde_mm256_castsi256_si128(v));
            return simde_mm_cvtepi32_ps(v32);
         }
         else if constexpr (CT::Integer32<FROM>) {
            // i/u32[4] -> float[4]                                     
            return simde_mm_cvtepi32_ps(simde_mm256_castsi256_si128(v));
         }
         else if constexpr (CT::SignedInteger64<FROM>) {
            // i64[4] -> float[4]                                       
            //TODO generalize this when 512 stuff is added to SIMDe     
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return simde_mm256_cvtepi64_ps(v);
            #else
               auto m1 = int64_to_double_full(simde_mm256_extracti128_si256(v, 0));
               auto m2 = int64_to_double_full(simde_mm256_extracti128_si256(v, 1));
               return simde_mm_movelh_ps(
                  simde_mm_cvtpd_ps(m1),
                  simde_mm_cvtpd_ps(m2)
               );
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<FROM>) {
            // u64[4] -> float[4]                                       
            //TODO generalize this when 512 stuff is added to SIMDe     
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return simde_mm256_cvtepu64_ps(v);
            #else
               auto m1 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 0));
               auto m2 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 1));
               return simde_mm_movelh_ps(
                  simde_mm_cvtpd_ps(m1),
                  simde_mm_cvtpd_ps(m2)
               );
            #endif
         }
         else LANGULUS_ERROR("Can't convert from __m256i to __m128");
      }
      else if constexpr (CT::SIMD128d<REGISTER>) {
         //                                                             
         // Converting TO double[2]                                     
         //                                                             
         if constexpr (CT::SignedInteger8<FROM>) {
            // i8[2] -> double[2]                                       
            const auto v32 = simde_mm_cvtepi8_epi32(simde_mm256_castsi256_si128(v));
            return simde_mm_cvtepi32_pd(v32);
         }
         else if constexpr (CT::UnsignedInteger8<FROM>) {
            // u8[2] -> double[2]                                       
            const auto v32 = simde_mm_cvtepu8_epi32(simde_mm256_castsi256_si128(v));
            return simde_mm_cvtepi32_pd(v32);
         }
         else if constexpr (CT::SignedInteger16<FROM>) {
            // i16[2] -> double[2]                                      
            const auto v32 = simde_mm_cvtepi16_epi32(simde_mm256_castsi256_si128(v));
            return simde_mm_cvtepi32_pd(v32);
         }
         else if constexpr (CT::UnsignedInteger16<FROM>) {
            // u16[2] -> double[2]                                      
            const auto v32 = simde_mm_cvtepu16_epi32(simde_mm256_castsi256_si128(v));
            return simde_mm_cvtepi32_pd(v32);
         }
         else if constexpr (CT::Integer32<FROM>) {
            // i/u32[2] -> double[2]                                    
            return simde_mm_cvtepi32_pd(simde_mm256_castsi256_si128(v));
         }
         else if constexpr (CT::SignedInteger64<FROM>) {
            // i64[2] -> double[2]                                      
            //TODO generalize this when 512 stuff is added to SIMDe     
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return simde_mm_cvtepi64_pd(simde_mm256_extracti128_si256(v, 0));
            #else
               return int64_to_double_full(simde_mm256_extracti128_si256(v, 0));
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<FROM>) {
            // u64[2] -> double[2]                                      
            //TODO generalize this when 512 stuff is added to SIMDe     
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return simde_mm_cvtepu64_pd(simde_mm256_extracti128_si256(v, 0));
            #else
               return uint64_to_double_full(simde_mm256_extracti128_si256(v, 0));
            #endif
         }
         else LANGULUS_ERROR("Can't convert from __m256i to __m128d");
      }
      else if constexpr (CT::SIMD128i<REGISTER>) {
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
         else LANGULUS_ERROR("Can't convert from __m256i to __m128d");
      }
      else LANGULUS_ERROR("Can't convert from __m256i to unsupported");
   }

} // namespace Langulus::SIMD