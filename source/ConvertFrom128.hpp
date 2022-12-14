///                                                                           
/// Langulus::SIMD                                                            
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>                    
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Load.hpp"

namespace Langulus::SIMD
{

   /// Convert __m128 to any other register                                   
   ///   @tparam TT - the true type contained in the result                   
   ///   @tparam S - size of the input array                                  
   ///   @tparam FT - true type contained in the input                        
   ///   @tparam TO - register to convert to                                  
   ///   @param v - the input data                                            
   ///   @return the resulting register                                       
   template<class TT, Count S, class FT, class TO>
   LANGULUS(ALWAYSINLINE) auto ConvertFrom128(const simde__m128& v) noexcept {
      //                                                                
      // Converting FROM float[4]                                       
      //                                                                
      if constexpr (CT::Same<TO, simde__m128d> && S <= 2) {
         //                                                             
         // Converting TO double[2]                                     
         //                                                             
         // float[2] -> double[2]                                       
         return simde_mm_cvtps_pd(v);
      }
      else if constexpr (CT::Same<TO, simde__m128i>) {
         //                                                             
         // Converting TO pci8[4], pcu8[4], pci16[4], pcu16[4]          
         //                 pci32[4], pcu32[4], pci64[2], pcu64[2]      
         //                                                             
         if constexpr (CT::SignedInteger8<TT> && S <= 4) {
            // float[4] -> pci8[4]                                      
            auto 
            vi32_16_8 = simde_mm_cvtps_epi32(v);
            vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::UnsignedInteger8<TT> && S <= 4) {
            // float[4] -> pcu8[4]                                      
            auto 
            vi32_16_8 = simde_mm_cvtps_epi32(v);
            vi32_16_8 = simde_mm_packus_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packus_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::SignedInteger16<TT> && S <= 4) {
            // float[4] -> pci16[4]                                     
            auto 
            vi32_16 = simde_mm_cvtps_epi32(v);
            vi32_16 = simde_mm_packs_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::UnsignedInteger16<TT> && S <= 4) {
            // float[4] -> pcu16[4]                                     
            auto 
            vi32_16 = simde_mm_cvtps_epi32(v);
            vi32_16 = simde_mm_packus_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::SignedInteger32<TT> && S <= 4) {
            // float[4] -> pci32[4]                                     
            return simde_mm_cvtps_epi32(v);
         }
         else if constexpr (CT::UnsignedInteger32<TT> && S <= 4) {
            // float[4] -> pcu32[4]                                     
            return _mm_cvtps_epu32(v);
         }
         else if constexpr (CT::SignedInteger64<TT> && S <= 2) {
            // float[2] -> pci64[2]                                     
            return _mm_cvtps_epi64(v);
         }
         else if constexpr (CT::UnsignedInteger64<TT> && S <= 2) {
            // float[2] -> pcu64[2]                                     
            return _mm_cvtps_epu64(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128 to __m128i");
      }
      else if constexpr (CT::Same<TO, simde__m256i>) {
         //                                                             
         // Converting TO pci64[4], pcu64[4]                            
         //                                                             
         if constexpr (CT::SignedInteger64<TT> && S <= 4) {
            // float[4] -> pci64[4]                                     
            return _mm256_cvtps_epi64(v);
         }
         else if constexpr (CT::UnsignedInteger64<TT> && S <= 4) {
            // float[4] -> pcu64[4]                                     
            return _mm256_cvtps_epu64(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128 to __m256i");
      }
      else if constexpr (CT::Same<TO, simde__m256d> && S <= 4) {
         //                                                             
         // Converting TO double[4]                                     
         //                                                             
         // float[4] -> double[4]                                       
         return simde_mm256_cvtps_pd(v);
      }
      else LANGULUS_ERROR("Can't convert from __m128 to unsupported");
   }

   /// Convert __m128d to any other register                                  
   ///   @tparam TT - the true type contained in the result                   
   ///   @tparam S - size of the input array                                  
   ///   @tparam FT - true type contained in the input                        
   ///   @tparam TO - register to convert to                                  
   ///   @param v - the input data                                            
   ///   @return the resulting register                                       
   template<class TT, Count S, class FT, class TO>
   LANGULUS(ALWAYSINLINE) auto ConvertFrom128d(const simde__m128d& v) noexcept {
      //                                                                
      // Converting FROM double[2]                                      
      //                                                                
      if constexpr (CT::Same<TO, simde__m128> && S <= 2) {
         //                                                             
         // Converting TO float[2]                                      
         //                                                             
         // double[2] -> float[2]                                       
         return simde_mm_cvtpd_ps(v);
      }
      else if constexpr (CT::Same<TO, simde__m128i>) {
         //                                                             
         // Converting TO pci8[2], pcu8[2], pci16[2], pcu16[2]          
         //                 pci32[2], pcu32[2], pci64[2], pcu64[2]      
         //                                                             
         if constexpr (CT::SignedInteger8<TT> && S <= 2) {
            // double[2] -> pci8[2]                                     
            auto 
            vi32_16_8 = simde_mm_cvtpd_epi32(v);
            vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::UnsignedInteger8<TT> && S <= 2) {
            // double[2] -> pcu8[2]                                     
            auto 
            vi32_16_8 = simde_mm_cvtpd_epi32(v);
            vi32_16_8 = simde_mm_packus_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packus_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::SignedInteger16<TT> && S <= 2) {
            // double[2] -> pci16[2]                                    
            auto 
            vi32_16 = simde_mm_cvtpd_epi32(v);
            vi32_16 = simde_mm_packs_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::UnsignedInteger16<TT> && S <= 2) {
            // double[2] -> pcu16[2]                                    
            auto 
            vi32_16 = simde_mm_cvtpd_epi32(v);
            vi32_16 = simde_mm_packus_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::Integer32<TT> && S <= 2) {
            // double[2] -> pci32[2] or pcu32[2]                        
            return simde_mm_cvtpd_pi32(v);
         }
         else if constexpr (CT::SignedInteger64<TT> && S <= 2) {
            // double[2] -> pci64[2] or pcu64[2]                        
            return _mm_cvtpd_epi64(v);
         }
         else if constexpr (CT::UnsignedInteger64<TT> && S <= 2) {
            // double[2] -> pci64[2] or pcu64[2]                        
            return _mm_cvtpd_epu64(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128d to __m128i");
      }
      else LANGULUS_ERROR("Can't convert from __m128d to unsupported");
   }

   /// Convert __m128i to any other register                                  
   ///   @tparam TT - the true type contained in the result                   
   ///   @tparam S - size of the input array                                  
   ///   @tparam FT - true type contained in the input                        
   ///   @tparam TO - register to convert to                                  
   ///   @param v - the input data                                            
   ///   @return the resulting register                                       
   template<class TT, Count S, class FT, class TO>
   LANGULUS(ALWAYSINLINE) auto ConvertFrom128i(const simde__m128i& v) noexcept {
      //                                                                
      // Converting FROM i/u8[16], i/u16[8], i/u32[4], i/u64[2]         
      //                                                                
      if constexpr (CT::Same<TO, simde__m128>) {
         //                                                             
         // Converting TO float[4] or float[2]                          
         //                                                             
         if constexpr (CT::SignedInteger8<FT> && S <= 4) {
            // i8[4] -> float[4]                                        
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_ps(vi16_32);                                 
         }
         else if constexpr (CT::UnsignedInteger8<FT> && S <= 4) {
            // u8[4] -> float[4]                                        
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return _mm_cvtepu32_ps(vi16_32);
         }
         else if constexpr (CT::SignedInteger16<FT> && S <= 4) {
            // i16[4] -> float[4]                                       
            auto 
            vi32 = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_ps(vi32);                                 
         }
         else if constexpr (CT::UnsignedInteger16<FT> && S <= 4) {
            // u16[4] -> float[4]                                       
            auto 
            cvt = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return _mm_cvtepu32_ps(cvt);
         }
         else if constexpr (CT::SignedInteger32<FT> && S <= 4) {
            // i32[4] -> float[4]                                       
            return simde_mm_cvtepi32_ps(v);
         }
         else if constexpr (CT::UnsignedInteger32<FT> && S <= 4) {
            // u32[4] -> float[4]                                       
            return simde_mm256_cvtpd_ps(simde_mm256_cvtepi32_pd(v));
         }
         else if constexpr (CT::SignedInteger64<FT> && S <= 2) {
            // i64[2] -> float[2]                                       
            return _mm_cvtepi64_ps(v);
         }
         else if constexpr (CT::UnsignedInteger64<FT> && S <= 2) {
            // u64[2] -> float[2]                                       
            return _mm_cvtepu64_ps(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128i to __m128");
      }
      else if constexpr (CT::Same<TO, simde__m128d>) {
         //                                                             
         // Converting TO double[2]                                     
         //                                                             
         if constexpr (CT::SignedInteger8<FT> && S <= 2) {
            // i8[2] -> double[2]                                       
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_pd(vi16_32);                                 
         }
         else if constexpr (CT::UnsignedInteger8<FT> && S <= 2) {
            // u8[2] -> double[2]                                       
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return _mm_cvtepu32_pd(vi16_32);                                 
         }
         else if constexpr (CT::SignedInteger16<FT> && S <= 2) {
            // i16[2] -> double[2]                                      
            auto 
            vi32 = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_pd(vi32);                                 
         }
         else if constexpr (CT::UnsignedInteger16<FT> && S <= 2) {
            // u16[2] -> double[2]                                      
            auto 
            vi32 = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return _mm_cvtepu32_pd(vi32);                                 
         }
         else if constexpr (CT::SignedInteger32<FT> && S <= 2) {
            // i32[2] -> double[2]                                      
            return simde_mm_cvtepi32_pd(v);
         }
         else if constexpr (CT::UnsignedInteger32<FT> && S <= 2) {
            // u32[2] -> double[2]                                      
            return _mm_cvtepu32_pd(v);
         }
         else if constexpr (CT::SignedInteger64<FT> && S <= 2) {
            // i64[2] -> double[2]                                      
            return _mm_cvtepi64_pd(v);
         }
         else if constexpr (CT::UnsignedInteger64<FT> && S <= 2) {
            // u64[2] -> double[2]                                      
            return _mm_cvtepu64_pd(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128i to __m128d");
      }
      else if constexpr (CT::Same<TO, simde__m128i>) {
         //                                                             
         // Converting TO i/u8[16], i/u16[8], i/u32[4], i/u64[2]        
         //                                                             
         if constexpr (CT::SignedInteger8<FT>) {
            if constexpr (CT::Integer8<TT> && S <= 16) {
               // i8[16] -> i/u8[16]                                    
               return v;
            }
            else if constexpr (CT::Integer16<TT> && S <= 8) {
               // i8[8] -> i/u16[8]                                     
               return simde_mm_cvtepi8_epi16(v);
            }
            else if constexpr (CT::Integer32<TT> && S <= 4) {
               // i8[4] -> i/u32[4]                                     
               return simde_mm_cvtepi8_epi32(v);
            }
            else if constexpr (CT::SignedInteger64<TT> && S <= 2) {
               // i8[2] -> i/u64[2]                                     
               return simde_mm_cvtepi8_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from i8 to unsupported TT");
         }
         else if constexpr (CT::UnsignedInteger8<FT>) {
            if constexpr (CT::Integer8<TT> && S <= 16) {
               // u8[16] -> i/u8[16]                                    
               return v;
            }
            else if constexpr (CT::Integer16<TT> && S <= 8) {
               // u8[8] -> i/u16[8]                                     
               return simde_mm_cvtepu8_epi16(v);
            }
            else if constexpr (CT::Integer32<TT> && S <= 4) {
               // u8[4] -> i/u32[4]                                     
               return simde_mm_cvtepu8_epi32(v);
            }
            else if constexpr (CT::SignedInteger64<TT> && S <= 2) {
               // u8[2] -> i/u64[2]                                     
               return simde_mm_cvtepu8_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from u8 to unsupported TT");
         }
         else if constexpr (CT::SignedInteger16<FT>) {
            if constexpr (CT::Integer8<TT> && S <= 8) {
               // i16[8] -> i/u8[8]                                     
               return lgls_pack_epi16(v, v);
            }
            else if constexpr (CT::Integer16<TT> && S <= 8) {
               // i16[8] -> i/u16[8]                                    
               return v;
            }
            else if constexpr (CT::Integer32<TT> && S <= 4) {
               // i16[4] -> i/u32[4]                                    
               return simde_mm_cvtepi16_epi32(v);
            }
            else if constexpr (CT::Integer64<TT> && S <= 2) {
               // i16[2] -> i/u64[2]                                    
               return simde_mm_cvtepi16_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from i16 to unsupported TT");
         }
         else if constexpr (CT::UnsignedInteger16<FT>) {
            if constexpr (CT::Integer8<TT> && S <= 8) {
               // u16[8] -> i/u8[8]                                     
               return lgls_pack_epi16(v, v);
            }
            else if constexpr (CT::Integer16<TT> && S <= 8) {
               // u16[8] -> i/u16[8]                                    
               return v;
            }
            else if constexpr (CT::Integer32<TT> && S <= 4) {
               // u16[4] -> i/u32[4]                                    
               return simde_mm_cvtepu16_epi32(v);
            }
            else if constexpr (CT::Integer64<TT> && S <= 2) {
               // u16[2] -> i/u64[2]                                    
               return simde_mm_cvtepu16_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from u16 to unsupported TT");
         }
         else if constexpr (CT::SignedInteger32<FT>) {
            if constexpr (CT::Integer8<TT> && S <= 4) {
               // i32[4] -> i/u8[4]                                     
               const auto t = lgls_pack_epi32(v, v);
               return lgls_pack_epi16(t, t);
            }
            else if constexpr (CT::Integer16<TT> && S <= 4) {
               // i32[4] -> i16[4]                                      
               return lgls_pack_epi32(v, v);
            }
            else if constexpr (CT::Integer32<TT> && S <= 4) {
               // i32[4] -> i32[4]                                      
               return v;
            }
            else if constexpr (CT::SignedInteger64<TT> && S <= 2) {
               // pci8[16] -> pcu8[16]                                  
               LANGULUS_ERROR("Can't convert from pci32[2] to pci64[2]");
            }
            else if constexpr (CT::UnsignedInteger64<TT> && S <= 2) {
               // pci8[16] -> pcu8[16]                                  
               LANGULUS_ERROR("Can't convert from pci32[2] to pcu64[2]");
            }
            else LANGULUS_ERROR("Can't convert from pci32 to unsupported TT");
         }
         else if constexpr (CT::UnsignedInteger32<FT>) {
            if constexpr (CT::SignedInteger8<TT> && S <= 4) {
               // pcu32[4] -> pci8[4]                                   
               return simde_mm_packus_epi16(simde_mm_packus_epi32(v, simde_mm_setzero_si128()), simde_mm_setzero_si128());
            }
            else if constexpr (CT::UnsignedInteger8<TT> && S <= 4) {
               // pcu32[4] -> pcu8[4]                                   
               return simde_mm_packus_epi16(simde_mm_packus_epi32(v, simde_mm_setzero_si128()), simde_mm_setzero_si128());
            }
            else if constexpr (CT::SignedInteger16<TT> && S <= 4) {
               // pcu32[4] -> pci16[4]                                  
               return simde_mm_packus_epi32(v, simde_mm_setzero_si128());
            }
            else if constexpr (CT::UnsignedInteger16<TT> && S <= 4) {
               // pcu32[4] -> pcu16[4]                                  
               return simde_mm_packus_epi32(v, simde_mm_setzero_si128());
            }
            else if constexpr (CT::SignedInteger32<TT> && S <= 4) {
               // pcu32[4] -> pci32[4]                                  
               auto lo = _mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(v));
               auto up = _mm_halfflip(_mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(_mm_halfflip(v))));
               return simde_mm_add_epi32(lo, up);
            }
            else if constexpr (CT::UnsignedInteger32<TT> && S <= 4) {
               // pcu32[4] -> pci32[4]                                  
               auto lo = _mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(v));
               auto up = _mm_halfflip(_mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(_mm_halfflip(v))));
               return simde_mm_add_epi32(lo, up);
            }
            else if constexpr (CT::SignedInteger64<TT> && S <= 2) {
               // pcu32[2] -> pci64[2]                                  
               return simde_mm_cvtepu32_epi64(v);
            }
            else if constexpr (CT::UnsignedInteger64<TT> && S <= 2) {
               // pcu32[2] -> pcu64[2]                                  
               return simde_mm_cvtepu32_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from pcu32 to unsupported TT");
         }
         else if constexpr (CT::SignedInteger64<FT>) {
            if constexpr (CT::SignedInteger8<TT> && S <= 2) {
               // pci64[2] -> pci8[2]                                   
               LANGULUS_ERROR("Can't convert from pci64[2] to pci8[2]");
            }
            else if constexpr (CT::UnsignedInteger8<TT> && S <= 2) {
               // pci64[2] -> pcu8[2]                                   
               LANGULUS_ERROR("Can't convert from pci64[2] to pcu8[2]");
            }
            else if constexpr (CT::SignedInteger16<TT> && S <= 2) {
               // pci64[2] -> pci16[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pci16[2]");
            }
            else if constexpr (CT::UnsignedInteger16<TT> && S <= 2) {
               // pci64[2] -> pcu16[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pcu16[2]");
            }
            else if constexpr (CT::SignedInteger32<TT> && S <= 2) {
               // pci64[2] -> pci32[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pci32[2]");
            }
            else if constexpr (CT::UnsignedInteger32<TT> && S <= 2) {
               // pci64[2] -> pcu32[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pcu32[2]");
            }
            else if constexpr (CT::SignedInteger64<TT> && S <= 2) {
               // pci64[2] -> pci64[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pci64[2]");
            }
            else if constexpr (CT::UnsignedInteger64<TT> && S <= 2) {
               // pci64[2] -> pcu64[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pcu64[2]");
            }
            else LANGULUS_ERROR("Can't convert from pci64 to unsupported TT");
         }
         else if constexpr (CT::UnsignedInteger64<FT>) {
            if constexpr (CT::SignedInteger8<TT> && S <= 2) {
               // pcu64[2] -> pci8[2]                                   
               LANGULUS_ERROR("Can't convert from pcu64[2] to pci8[2]");
            }
            else if constexpr (CT::UnsignedInteger8<TT> && S <= 2) {
               // pcu64[2] -> pcu8[2]                                   
               LANGULUS_ERROR("Can't convert from pcu64[2] to pcu8[2]");
            }
            else if constexpr (CT::SignedInteger16<TT> && S <= 2) {
               // pcu64[2] -> pci16[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pci16[2]");
            }
            else if constexpr (CT::UnsignedInteger16<TT> && S <= 2) {
               // pcu64[2] -> pcu16[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pcu16[2]");
            }
            else if constexpr (CT::SignedInteger32<TT> && S <= 2) {
               // pcu64[2] -> pci32[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pci32[2]");
            }
            else if constexpr (CT::UnsignedInteger32<TT> && S <= 2) {
               // pcu64[2] -> pcu32[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pcu32[2]");
            }
            else if constexpr (CT::SignedInteger64<TT> && S <= 2) {
               // pcu64[2] -> pci64[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pci64[2]");
            }
            else if constexpr (CT::UnsignedInteger64<TT> && S <= 2) {
               // pcu64[2] -> pcu64[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pcu64[2]");
            }
            else LANGULUS_ERROR("Can't convert from pcu64 to unsupported TT");
         }
         else LANGULUS_ERROR("Can't convert from __m128i to __m128i");
      }
      else if constexpr (CT::Same<TO, __m128d>) {
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