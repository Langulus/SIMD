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
   LANGULUS(INLINED)
   auto ConvertFrom128(const simde__m128& v) noexcept {
      //                                                                
      // Converting FROM float[4]                                       
      //                                                                
      if constexpr (CT::Same<TO, simde__m128d> and S <= 2) {
         //                                                             
         // Converting TO double[2]                                     
         //                                                             
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[2] to double[2]");
         return simde_mm_cvtps_pd(v);
      }
      else if constexpr (CT::Same<TO, simde__m128i>) {
         //                                                             
         // Converting TO pci8[4], pcu8[4], pci16[4], pcu16[4]          
         //                 pci32[4], pcu32[4], pci64[2], pcu64[2]      
         //                                                             
         if constexpr (CT::SignedInteger8<TT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to int8[4]");
            auto 
            vi32_16_8 = simde_mm_cvtps_epi32(v);
            vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::UnsignedInteger8<TT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to uint8[4]");
            auto
            vi32_16_8 = simde_mm_cvtps_epi32(v);
            vi32_16_8 = simde_mm_packus_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packus_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::SignedInteger16<TT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to int16[4]");
            auto
            vi32_16 = simde_mm_cvtps_epi32(v);
            vi32_16 = simde_mm_packs_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::UnsignedInteger16<TT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to uint16[4]");
            auto
            vi32_16 = simde_mm_cvtps_epi32(v);
            vi32_16 = simde_mm_packus_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::SignedInteger32<TT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to int32[4]");
            return simde_mm_cvtps_epi32(v);
         }
         else if constexpr (CT::UnsignedInteger32<TT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[4] to uint32[4]");
            return _mm_cvtps_epu32(v);
         }
         else if constexpr (CT::SignedInteger64<TT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[2] to int64[2]");
            return _mm_cvtps_epi64(v);
         }
         else if constexpr (CT::UnsignedInteger64<TT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from float[2] to uint64[2]");
            return _mm_cvtps_epu64(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128 to __m128i");
      }
      else if constexpr (CT::Same<TO, simde__m256i>) {
         //                                                             
         // Converting TO pci64[4], pcu64[4]                            
         //                                                             
         if constexpr (CT::SignedInteger64<TT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 256bit register from float[4] to int64[4]");
            return _mm256_cvtps_epi64(v);
         }
         else if constexpr (CT::UnsignedInteger64<TT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 256bit register from float[4] to uint64[4]");
            return _mm256_cvtps_epu64(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128 to __m256i");
      }
      else if constexpr (CT::Same<TO, simde__m256d> and S <= 4) {
         //                                                             
         // Converting TO double[4]                                     
         //                                                             
         LANGULUS_SIMD_VERBOSE("Converting 256bit register from float[4] to double[4]");
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
   LANGULUS(INLINED)
   auto ConvertFrom128d(const simde__m128d& v) noexcept {
      //                                                                
      // Converting FROM double[2]                                      
      //                                                                
      if constexpr (CT::Same<TO, simde__m128> and S <= 2) {
         //                                                             
         // Converting TO float[2]                                      
         //                                                             
         LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to float[2]");
         return simde_mm_cvtpd_ps(v);
      }
      else if constexpr (CT::Same<TO, simde__m128i>) {
         //                                                             
         // Converting TO pci8[2], pcu8[2], pci16[2], pcu16[2]          
         //                 pci32[2], pcu32[2], pci64[2], pcu64[2]      
         //                                                             
         if constexpr (CT::SignedInteger8<TT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to int8[2]");
            auto
            vi32_16_8 = simde_mm_cvtpd_epi32(v);
            vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::UnsignedInteger8<TT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to uint8[2]");
            auto
            vi32_16_8 = simde_mm_cvtpd_epi32(v);
            vi32_16_8 = simde_mm_packus_epi32(vi32_16_8, simde_mm_setzero_si128());
            vi32_16_8 = simde_mm_packus_epi16(vi32_16_8, simde_mm_setzero_si128());
            return vi32_16_8;
         }
         else if constexpr (CT::SignedInteger16<TT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to int16[2]");
            auto
            vi32_16 = simde_mm_cvtpd_epi32(v);
            vi32_16 = simde_mm_packs_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::UnsignedInteger16<TT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to uint16[2]");
            auto
            vi32_16 = simde_mm_cvtpd_epi32(v);
            vi32_16 = simde_mm_packus_epi32(vi32_16, simde_mm_setzero_si128());
            return vi32_16;
         }
         else if constexpr (CT::Integer32<TT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to int32/uint32[2]");
            return simde_mm_cvtpd_pi32(v);
         }
         else if constexpr (CT::SignedInteger64<TT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to int64[2]");
            return _mm_cvtpd_epi64(v);
         }
         else if constexpr (CT::UnsignedInteger64<TT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from double[2] to uint64[2]");
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
   LANGULUS(INLINED)
   auto ConvertFrom128i(const simde__m128i& v) noexcept {
      //                                                                
      // Converting FROM i/u8[16], i/u16[8], i/u32[4], i/u64[2]         
      //                                                                
      if constexpr (CT::Same<TO, simde__m128>) {
         //                                                             
         // Converting TO float[4] or float[2]                          
         //                                                             
         if constexpr (CT::SignedInteger8<FT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from int8[4] to float[4]");
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_ps(vi16_32);                                 
         }
         else if constexpr (CT::UnsignedInteger8<FT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint8[4] to float[4]");
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return _mm_cvtepu32_ps(vi16_32);
         }
         else if constexpr (CT::SignedInteger16<FT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from int16[4] to float[4]");
            auto
            vi32 = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_ps(vi32);                                 
         }
         else if constexpr (CT::UnsignedInteger16<FT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint16[4] to float[4]");
            auto
            cvt = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return _mm_cvtepu32_ps(cvt);
         }
         else if constexpr (CT::SignedInteger32<FT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from int32[4] to float[4]");
            return simde_mm_cvtepi32_ps(v);
         }
         else if constexpr (CT::UnsignedInteger32<FT> and S <= 4) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint32[4] to float[4]");
            return simde_mm256_cvtpd_ps(simde_mm256_cvtepi32_pd(v));
         }
         else if constexpr (CT::SignedInteger64<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from int64[2] to float[2]");
            return _mm_cvtepi64_ps(v);
         }
         else if constexpr (CT::UnsignedInteger64<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint64[2] to float[2]");
            return _mm_cvtepu64_ps(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128i to __m128");
      }
      else if constexpr (CT::Same<TO, simde__m128d>) {
         //                                                             
         // Converting TO double[2]                                     
         //                                                             
         if constexpr (CT::SignedInteger8<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from int8[2] to double[2]");
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_pd(vi16_32);                                 
         }
         else if constexpr (CT::UnsignedInteger8<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint8[2] to double[2]");
            auto
            vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
            vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
            return _mm_cvtepu32_pd(vi16_32);                                 
         }
         else if constexpr (CT::SignedInteger16<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from int16[2] to double[2]");
            auto
            vi32 = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return simde_mm_cvtepi32_pd(vi32);                                 
         }
         else if constexpr (CT::UnsignedInteger16<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint16[2] to double[2]");
            auto
            vi32 = simde_mm_unpacklo_epi16(v, simde_mm_setzero_si128());
            return _mm_cvtepu32_pd(vi32);                                 
         }
         else if constexpr (CT::SignedInteger32<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from int32[2] to double[2]");
            return simde_mm_cvtepi32_pd(v);
         }
         else if constexpr (CT::UnsignedInteger32<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint32[2] to double[2]");
            return _mm_cvtepu32_pd(v);
         }
         else if constexpr (CT::SignedInteger64<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from int64[2] to double[2]");
            return _mm_cvtepi64_pd(v);
         }
         else if constexpr (CT::UnsignedInteger64<FT> and S <= 2) {
            LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint64[2] to double[2]");
            return _mm_cvtepu64_pd(v);
         }
         else LANGULUS_ERROR("Can't convert from __m128i to __m128d");
      }
      else if constexpr (CT::Same<TO, simde__m128i>) {
         //                                                             
         // Converting TO i/u8[16], i/u16[8], i/u32[4], i/u64[2]        
         //                                                             
         if constexpr (CT::SignedInteger8<FT>) {
            if constexpr (CT::Integer8<TT> and S <= 16) {
               return v;
            }
            else if constexpr (CT::Integer16<TT> and S <= 8) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from int8[8] to int16/uint16[8]");
               return simde_mm_cvtepi8_epi16(v);
            }
            else if constexpr (CT::Integer32<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from int8[4] to int32/uint32[4]");
               return simde_mm_cvtepi8_epi32(v);
            }
            else if constexpr (CT::SignedInteger64<TT> and S <= 2) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from int8[2] to int64[2]");
               return simde_mm_cvtepi8_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from i8 to unsupported TT");
         }
         else if constexpr (CT::UnsignedInteger8<FT>) {
            if constexpr (CT::Integer8<TT> and S <= 16) {
               return v;
            }
            else if constexpr (CT::Integer16<TT> and S <= 8) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint8[8] to int16/uint16[8]");
               return simde_mm_cvtepu8_epi16(v);
            }
            else if constexpr (CT::Integer32<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint8[4] to int32/uint32[4]");
               return simde_mm_cvtepu8_epi32(v);
            }
            else if constexpr (CT::SignedInteger64<TT> and S <= 2) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint8[2] to int64[2]");
               return simde_mm_cvtepu8_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from u8 to unsupported TT");
         }
         else if constexpr (CT::SignedInteger16<FT>) {
            if constexpr (CT::Integer8<TT> and S <= 8) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from int16[8] to int8/uint8[8]");
               return lgls_pack_epi16(v, v);
            }
            else if constexpr (CT::Integer16<TT> and S <= 8) {
               return v;
            }
            else if constexpr (CT::Integer32<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from int16[4] to int32/uint32[4]");
               return simde_mm_cvtepi16_epi32(v);
            }
            else if constexpr (CT::Integer64<TT> and S <= 2) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from int16[2] to int64/uint64[2]");
               return simde_mm_cvtepi16_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from i16 to unsupported TT");
         }
         else if constexpr (CT::UnsignedInteger16<FT>) {
            if constexpr (CT::Integer8<TT> and S <= 8) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint16[8] to int8/uint8[8]");
               return lgls_pack_epi16(v, v);
            }
            else if constexpr (CT::Integer16<TT> and S <= 8) {
               return v;
            }
            else if constexpr (CT::Integer32<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint16[4] to int32/uint32[4]");
               return simde_mm_cvtepu16_epi32(v);
            }
            else if constexpr (CT::Integer64<TT> and S <= 2) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint16[2] to int64/uint64[2]");
               return simde_mm_cvtepu16_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from u16 to unsupported TT");
         }
         else if constexpr (CT::SignedInteger32<FT>) {
            if constexpr (CT::Integer8<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from int32[4] to int8/uint8[4]");
               const auto t = lgls_pack_epi32(v, v);
               return lgls_pack_epi16(t, t);
            }
            else if constexpr (CT::Integer16<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from int32[4] to int16/uint16[4]");
               return lgls_pack_epi32(v, v);
            }
            else if constexpr (CT::Integer32<TT> and S <= 4) {
               return v;
            }
            else if constexpr (CT::SignedInteger64<TT> and S <= 2) {
               LANGULUS_ERROR("Can't convert from pci32[2] to pci64[2]");
            }
            else if constexpr (CT::UnsignedInteger64<TT> and S <= 2) {
               LANGULUS_ERROR("Can't convert from pci32[2] to pcu64[2]");
            }
            else LANGULUS_ERROR("Can't convert from pci32 to unsupported TT");
         }
         else if constexpr (CT::UnsignedInteger32<FT>) {
            if constexpr (CT::SignedInteger8<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint32[4] to int8[4]");
               return simde_mm_packus_epi16(simde_mm_packus_epi32(v, simde_mm_setzero_si128()), simde_mm_setzero_si128());
            }
            else if constexpr (CT::UnsignedInteger8<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint32[4] to uint8[4]");
               return simde_mm_packus_epi16(simde_mm_packus_epi32(v, simde_mm_setzero_si128()), simde_mm_setzero_si128());
            }
            else if constexpr (CT::SignedInteger16<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint32[4] to int16[4]");
               return simde_mm_packus_epi32(v, simde_mm_setzero_si128());
            }
            else if constexpr (CT::UnsignedInteger16<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint32[4] to uint16[4]");
               return simde_mm_packus_epi32(v, simde_mm_setzero_si128());
            }
            else if constexpr (CT::SignedInteger32<TT> and S <= 4) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint32[4] to int32[4]");
               auto lo = _mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(v));
               auto up = _mm_halfflip(_mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(_mm_halfflip(v))));
               return simde_mm_add_epi32(lo, up);
            }
            else if constexpr (CT::UnsignedInteger32<TT> and S <= 4) {
               return v;
            }
            else if constexpr (CT::SignedInteger64<TT> and S <= 2) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint32[2] to int64[2]");
               return simde_mm_cvtepu32_epi64(v);
            }
            else if constexpr (CT::UnsignedInteger64<TT> and S <= 2) {
               LANGULUS_SIMD_VERBOSE("Converting 128bit register from uint32[2] to uint64[2]");
               return simde_mm_cvtepu32_epi64(v);
            }
            else LANGULUS_ERROR("Can't convert from pcu32 to unsupported TT");
         }
         else if constexpr (CT::SignedInteger64<FT>) {
            if constexpr (CT::SignedInteger8<TT> and S <= 2) {
               // pci64[2] -> pci8[2]                                   
               LANGULUS_ERROR("Can't convert from pci64[2] to pci8[2]");
            }
            else if constexpr (CT::UnsignedInteger8<TT> and S <= 2) {
               // pci64[2] -> pcu8[2]                                   
               LANGULUS_ERROR("Can't convert from pci64[2] to pcu8[2]");
            }
            else if constexpr (CT::SignedInteger16<TT> and S <= 2) {
               // pci64[2] -> pci16[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pci16[2]");
            }
            else if constexpr (CT::UnsignedInteger16<TT> and S <= 2) {
               // pci64[2] -> pcu16[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pcu16[2]");
            }
            else if constexpr (CT::SignedInteger32<TT> and S <= 2) {
               // pci64[2] -> pci32[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pci32[2]");
            }
            else if constexpr (CT::UnsignedInteger32<TT> and S <= 2) {
               // pci64[2] -> pcu32[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pcu32[2]");
            }
            else if constexpr (CT::SignedInteger64<TT> and S <= 2) {
               // pci64[2] -> pci64[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pci64[2]");
            }
            else if constexpr (CT::UnsignedInteger64<TT> and S <= 2) {
               // pci64[2] -> pcu64[2]                                  
               LANGULUS_ERROR("Can't convert from pci64[2] to pcu64[2]");
            }
            else LANGULUS_ERROR("Can't convert from pci64 to unsupported TT");
         }
         else if constexpr (CT::UnsignedInteger64<FT>) {
            if constexpr (CT::SignedInteger8<TT> and S <= 2) {
               // pcu64[2] -> pci8[2]                                   
               LANGULUS_ERROR("Can't convert from pcu64[2] to pci8[2]");
            }
            else if constexpr (CT::UnsignedInteger8<TT> and S <= 2) {
               // pcu64[2] -> pcu8[2]                                   
               LANGULUS_ERROR("Can't convert from pcu64[2] to pcu8[2]");
            }
            else if constexpr (CT::SignedInteger16<TT> and S <= 2) {
               // pcu64[2] -> pci16[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pci16[2]");
            }
            else if constexpr (CT::UnsignedInteger16<TT> and S <= 2) {
               // pcu64[2] -> pcu16[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pcu16[2]");
            }
            else if constexpr (CT::SignedInteger32<TT> and S <= 2) {
               // pcu64[2] -> pci32[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pci32[2]");
            }
            else if constexpr (CT::UnsignedInteger32<TT> and S <= 2) {
               // pcu64[2] -> pcu32[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pcu32[2]");
            }
            else if constexpr (CT::SignedInteger64<TT> and S <= 2) {
               // pcu64[2] -> pci64[2]                                  
               LANGULUS_ERROR("Can't convert from pcu64[2] to pci64[2]");
            }
            else if constexpr (CT::UnsignedInteger64<TT> and S <= 2) {
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