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
   
   /// https://stackoverflow.com/questions/41144668                           
   //  Only works for inputs in the range: [0, 2^52)
   LANGULUS(INLINED)
   simde__m128d uint64_to_double(simde__m128i x) noexcept {
      x = simde_mm_or_si128(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)));
      return simde_mm_sub_pd(simde_mm_castsi128_pd(x), simde_mm_set1_pd(0x0010000000000000));
   }

   //  Only works for inputs in the range: [-2^51, 2^51]
   LANGULUS(INLINED)
   simde__m128d int64_to_double(simde__m128i x) noexcept {
      x = simde_mm_add_epi64(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0018000000000000)));
      return simde_mm_sub_pd(simde_mm_castsi128_pd(x), simde_mm_set1_pd(0x0018000000000000));
   }

   LANGULUS(INLINED)
   simde__m128d uint64_to_double_full(simde__m128i x) noexcept {
      simde__m128i xH = simde_mm_srli_epi64(x, 32);
      xH = simde_mm_or_si128(xH, simde_mm_castpd_si128(simde_mm_set1_pd(19342813113834066795298816.)));          //  2^84
      simde__m128i xL = simde_mm_blend_epi16(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
      simde__m128d f = simde_mm_sub_pd(simde_mm_castsi128_pd(xH), simde_mm_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52
      return simde_mm_add_pd(f, simde_mm_castsi128_pd(xL));
   }

   LANGULUS(INLINED)
   simde__m128d int64_to_double_full(simde__m128i x) noexcept {
      simde__m128i xH = simde_mm_srai_epi32(x, 16);
      xH = simde_mm_blend_epi16(xH, simde_mm_setzero_si128(), 0x33);
      xH = simde_mm_add_epi64(xH, simde_mm_castpd_si128(simde_mm_set1_pd(442721857769029238784.)));              //  3*2^67
      simde__m128i xL = simde_mm_blend_epi16(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)), 0x88);   //  2^52
      simde__m128d f = simde_mm_sub_pd(simde_mm_castsi128_pd(xH), simde_mm_set1_pd(442726361368656609280.));          //  3*2^67 + 2^52
      return simde_mm_add_pd(f, simde_mm_castsi128_pd(xL));
   }


   /// Convert V128i to any other register                                    
   ///   @tparam TO - the desired element type                                
   ///   @param v - the input register                                        
   ///   @return the converted register                                       
   template<Element TO> LANGULUS(INLINED)
   auto ConvertFrom128i(CT::SIMD128i auto v) noexcept {
      using R = decltype(v);
      using T = TypeOf<R>;

      if constexpr (CT::Double<TO>) {
         //                                                             
         // Converting as many doubles as possible                      
         //                                                             
         if constexpr (CT::SignedInteger8<T>) {
            // i8[4] -> double[4]                                       
            LANGULUS_SIMD_VERBOSE("Converting signed 8bit ints -> 64bit floats");
            const auto v32 = simde_mm_cvtepi8_epi32(v);
            return V128<TO> {simde_mm_cvtepi32_pd(v32)};
         }
         else if constexpr (CT::UnsignedInteger8<T>) {
            // u8[4] -> double[4]                                       
            LANGULUS_SIMD_VERBOSE("Converting unsigned 8bit ints -> 64bit floats");
            const auto v32 = simde_mm_cvtepu8_epi32(v);
            return V128<TO> {simde_mm_cvtepi32_pd(v32)};
         }
         else if constexpr (CT::SignedInteger16<T>) {
            // i16[4] -> double[4]                                      
            LANGULUS_SIMD_VERBOSE("Converting signed 16bit ints -> 64bit floats");
            const auto v32 = simde_mm_cvtepi16_epi32(v);
            return V128<TO> {simde_mm_cvtepi32_pd(v32)};
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            // u16[4] -> double[4]                                      
            LANGULUS_SIMD_VERBOSE("Converting unsigned 16bit ints -> 64bit floats");
            const auto v32 = simde_mm_cvtepu16_epi32(v);
            return V128<TO> {simde_mm_cvtepi32_pd(v32)};
         }
         else if constexpr (CT::Integer32<T>) {
            // i/u32[4] -> double[4]                                    
            LANGULUS_SIMD_VERBOSE("Converting 32bit ints -> 64bit floats");
            return V128<TO> {simde_mm_cvtepi32_pd(v)};
         }
         else if constexpr (CT::SignedInteger64<T>) {
            // i64[4] -> double[4]                                      
            //TODO generalize this when 512 stuff is added to SIMDe     
            LANGULUS_SIMD_VERBOSE("Converting signed 64bit ints -> 64bit floats");
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return V128<TO> {simde_mm_cvtepi64_pd(v)};
            #elif LANGULUS_SIMD(256BIT)
               return V128<TO> {int64_to_double_full(v)};
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<T>) {
            // u64[4] -> double[4]                                      
            //TODO generalize this when 512 stuff is added to SIMDe     
            LANGULUS_SIMD_VERBOSE("Converting unsigned 64bit ints -> 64bit floats");
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return V128<TO> {simde_mm_cvtepu64_pd(v)};
            #else
               return V128<TO> {uint64_to_double_full(v)};
            #endif
         }
         else static_assert(false, "Unsupported conversion");
      }
      else if constexpr (CT::Float<TO>) {
         //                                                             
         // Converting to floats                                        
         //                                                             
         if constexpr (CT::SignedInteger8<T>) {
            // i8[8] -> float[8]                                        
            LANGULUS_SIMD_VERBOSE("Converting signed 8bit ints -> 32bit floats");
            const auto v32 = simde_mm_cvtepi8_epi32(v);
            return V128<TO> {simde_mm_cvtepi32_ps(v32)};
         }
         else if constexpr (CT::UnsignedInteger8<T>) {
            // u8[8] -> float[8]                                        
            LANGULUS_SIMD_VERBOSE("Converting unsigned 8bit ints -> 32bit floats");
            const auto v32 = simde_mm_cvtepu8_epi32(v);
            return V128<TO> {simde_mm_cvtepi32_ps(v32)};
         }
         else if constexpr (CT::SignedInteger16<T>) {
            // i16[8] -> float[8]                                       
            LANGULUS_SIMD_VERBOSE("Converting signed 16bit ints -> 32bit floats");
            const auto v32 = simde_mm_cvtepi16_epi32(v);
            return V128<TO> {simde_mm_cvtepi32_ps(v32)};
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            // u16[8] -> float[8]                                       
            LANGULUS_SIMD_VERBOSE("Converting unsigned 16bit ints -> 32bit floats");
            const auto v32 = simde_mm_cvtepu16_epi32(v);
            return V128<TO> {simde_mm_cvtepi32_ps(v32)};
         }
         else if constexpr (CT::Integer32<T>) {
            // i/u32[8] -> float[8]                                     
            LANGULUS_SIMD_VERBOSE("Converting 32bit ints -> 32bit floats");
            return V128<TO> {simde_mm_cvtepi32_ps(v)};
         }
         else if constexpr (CT::SignedInteger64<T>) {
            // i64[4] -> float[4]                                       
            //TODO generalize this when 512 stuff is added to SIMDe     
            LANGULUS_SIMD_VERBOSE("Converting signed 64bit ints -> 32bit floats");
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return V128<TO> {simde_mm_cvtepi64_ps(v)};
            #else
               const auto m1 = int64_to_double_full(v);
               return V128<TO> {simde_mm_cvtpd_ps(m1)};
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<T>) {
            // u64[4] -> float[4]                                       
            //TODO generalize this when 512 stuff is added to SIMDe     
            LANGULUS_SIMD_VERBOSE("Converting unsigned 64bit ints -> 32bit floats");
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return V128<TO> {simde_mm_cvtepu64_ps(v)};
            #else
               const auto m1 = uint64_to_double_full(v);
               return V128<TO> {simde_mm_cvtpd_ps(m1)};
            #endif
         }
         else static_assert(false, "Unsupported conversion");
      }
      else if constexpr (CT::Integer8<TO>) {
         //                                                             
         // Converting to 8bit integer                                  
         //                                                             
         if constexpr (CT::Integer8<T>) {
            LANGULUS_SIMD_VERBOSE("No conversion required");
            return V128<TO> {v};
         }
         else if constexpr (CT::Integer16<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 16bit ints -> 8bit ints");
            return V128<TO> {v.Pack()};
         }
         else if constexpr (CT::Integer32<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 32bit ints -> 8bit ints");
            return V128<TO> {v.Pack().Pack()};
         }
         else if constexpr (CT::Integer64<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 64bit ints -> 8bit ints");
            return V128<TO> {v.Pack().Pack().Pack()};
         }
         else static_assert(false, "Unsupported conversion");
      }
      else if constexpr (CT::Integer16<TO>) {
         //                                                             
         // Converting to 16bit integer                                 
         //                                                             
         if constexpr (CT::Integer8<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 8bit ints -> 16bit ints");
            return V128<TO> {v.UnpackLo()};
         }
         else if constexpr (CT::Integer16<T>) {
            LANGULUS_SIMD_VERBOSE("No conversion required");
            return V128<TO> {v};
         }
         else if constexpr (CT::Integer32<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 32bit ints -> 16bit ints");
            return V128<TO> {v.Pack()};
         }
         else if constexpr (CT::Integer64<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 64bit ints -> 16bit ints");
            return V128<TO> {v.Pack().Pack()};
         }
         else static_assert(false, "Unsupported conversion");
      }
      else if constexpr (CT::Integer32<TO>) {
         //                                                             
         // Converting to 32bit integer                                 
         //                                                             
         if constexpr (CT::Integer8<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 8bit ints -> 32bit ints");
            return V128<TO> {v.UnpackLo().UnpackLo()};
         }
         else if constexpr (CT::Integer16<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 16bit ints -> 32bit ints");
            return V128<TO> {v.UnpackLo()};
         }
         else if constexpr (CT::Integer32<T>) {
            LANGULUS_SIMD_VERBOSE("No conversion required");
            return V128<TO> {v};
         }
         else if constexpr (CT::Integer64<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 64bit ints -> 32bit ints");
            return V128<TO> {v.Pack()};
         }
         else static_assert(false, "Unsupported conversion");
      }
      else if constexpr (CT::Integer64<TO>) {
         //                                                             
         // Converting to 64bit integer                                 
         //                                                             
         if constexpr (CT::Integer8<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 8bit ints -> 64bit ints");
            return V128<TO> {v.UnpackLo().UnpackLo().UnpackLo()};
         }
         else if constexpr (CT::Integer16<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 16bit ints -> 64bit ints");
            return V128<TO> {v.UnpackLo().UnpackLo()};
         }
         else if constexpr (CT::Integer32<T>) {
            LANGULUS_SIMD_VERBOSE("Converting 32bit ints -> 64bit ints");
            return V128<TO> {v.UnpackLo()};
         }
         else if constexpr (CT::Integer64<T>) {
            LANGULUS_SIMD_VERBOSE("No conversion required");
            return V128<TO> {v};
         }
         else static_assert(false, "Unsupported conversion");
      }
      else static_assert(false, "Unsupported register");
   }

} // namespace Langulus::SIMD