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

   /// Convert V256i to any other register                                    
   ///   @tparam TO - the desired element type                                
   ///   @param v - the input register                                        
   ///   @return the converted register                                       
   template<Element TO> NOD() LANGULUS(INLINED)
   auto ConvertFrom256i(CT::SIMD256i auto v) noexcept {
      using R = decltype(v);
      using T = TypeOf<R>;

      if constexpr (CT::Double<TO>) {
         //                                                             
         // Converting as many doubles as possible                      
         //                                                             
         if constexpr (CT::SignedInteger8<T>) {
            // i8[4] -> double[4]                                       
            const auto v32 = simde_mm256_cvtepi8_epi32(simde_mm256_castsi256_si128(v));
            return V256<TO> {simde_mm256_cvtepi32_pd(simde_mm256_castsi256_si128(v32))};
         }
         else if constexpr (CT::UnsignedInteger8<T>) {
            // u8[4] -> double[4]                                       
            const auto v32 = simde_mm256_cvtepu8_epi32(simde_mm256_castsi256_si128(v));
            return V256<TO> {simde_mm256_cvtepi32_pd(simde_mm256_castsi256_si128(v32))};
         }
         else if constexpr (CT::SignedInteger16<T>) {
            // i16[4] -> double[4]                                      
            const auto v32 = simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(v));
            return V256<TO> {simde_mm256_cvtepi32_pd(simde_mm256_castsi256_si128(v32))};
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            // u16[4] -> double[4]                                      
            const auto v32 = simde_mm256_cvtepu16_epi32(simde_mm256_castsi256_si128(v));
            return V256<TO> {simde_mm256_cvtepi32_pd(simde_mm256_castsi256_si128(v32))};
         }
         else if constexpr (CT::Integer32<T>) {
            // i/u32[4] -> double[4]                                    
            return V256<TO> {simde_mm256_cvtepi32_pd(simde_mm256_castsi256_si128(v))};
         }
         else if constexpr (CT::SignedInteger64<T>) {
            // i64[4] -> double[4]                                      
            //TODO generalize this when 512 stuff is added to SIMDe     
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return V256<TO> {simde_mm256_cvtepi64_pd(v)};
            #elif LANGULUS_SIMD(256BIT)
               auto m1 = int64_to_double_full(simde_mm256_extracti128_si256(v, 0));
               auto m2 = int64_to_double_full(simde_mm256_extracti128_si256(v, 1));
               return V256<TO> {simde_mm256_set_m128d(m1, m2)};
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<T>) {
            // u64[4] -> double[4]                                      
            //TODO generalize this when 512 stuff is added to SIMDe     
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return V256<TO> {simde_mm_cvtepu64_pd(simde_mm256_extracti128_si256(v, 0))};
            #else
               auto m1 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 0));
               auto m2 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 1));
               return V256<TO> {simde_mm256_set_m128d(m1, m2)};
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
            const auto v32 = simde_mm256_cvtepi8_epi32(simde_mm256_castsi256_si128(v));
            return V256<TO> {simde_mm256_cvtepi32_ps(v32)};
         }
         else if constexpr (CT::UnsignedInteger8<T>) {
            // u8[8] -> float[8]                                        
            const auto v32 = simde_mm256_cvtepu8_epi32(simde_mm256_castsi256_si128(v));
            return V256<TO> {simde_mm256_cvtepi32_ps(v32)};
         }
         else if constexpr (CT::SignedInteger16<T>) {
            // i16[8] -> float[8]                                       
            const auto v32 = simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(v));
            return V256<TO> {simde_mm256_cvtepi32_ps(v32)};
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            // u16[8] -> float[8]                                       
            const auto v32 = simde_mm256_cvtepu16_epi32(simde_mm256_castsi256_si128(v));
            return V256<TO> {simde_mm256_cvtepi32_ps(v32)};
         }
         else if constexpr (CT::Integer32<T>) {
            // i/u32[8] -> float[8]                                     
            return V256<TO> {simde_mm256_cvtepi32_ps(v)};
         }
         else if constexpr (CT::SignedInteger64<T>) {
            // i64[4] -> float[4]                                       
            //TODO generalize this when 512 stuff is added to SIMDe     
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return V256<TO> {simde_mm256_cvtepi64_ps(v)};
            #else
               auto m1 = int64_to_double_full(simde_mm256_extracti128_si256(v, 0));
               auto m2 = int64_to_double_full(simde_mm256_extracti128_si256(v, 1));
               return V256<TO> {simde_mm256_set_m128(
                  simde_mm_movelh_ps(simde_mm_cvtpd_ps(m1), simde_mm_cvtpd_ps(m2)),
                  simde_mm_setzero_ps()
               )};
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<T>) {
            // u64[4] -> float[4]                                       
            //TODO generalize this when 512 stuff is added to SIMDe     
            #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
               return V256<TO> {simde_mm256_cvtepu64_ps(v)};
            #else
               auto m1 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 0));
               auto m2 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 1));
               return V256<TO> {simde_mm256_set_m128(
                  simde_mm_movelh_ps(simde_mm_cvtpd_ps(m1), simde_mm_cvtpd_ps(m2)),
                  simde_mm_setzero_ps()
               )};
            #endif
         }
         else static_assert(false, "Unsupported conversion");
      }
      else if constexpr (CT::Integer8<TO>) {
         //                                                             
         // Converting to 8bit integer                                  
         //                                                             
         if constexpr (CT::Integer8<T>)
            return V256<TO> {v};
         else if constexpr (CT::Integer16<T>)
            return V256<TO> {v.Pack()};
         else if constexpr (CT::Integer32<T>)
            return V256<TO> {v.Pack().Pack()};
         else if constexpr (CT::Integer64<T>)
            return V256<TO> {v.Pack().Pack().Pack()};
         else
            static_assert(false, "Unsupported conversion");
      }
      else if constexpr (CT::Integer16<TO>) {
         //                                                             
         // Converting to 16bit integer                                 
         //                                                             
         if constexpr (CT::Integer8<T>)
            return V256<TO> {v.UnpackLo()};
         else if constexpr (CT::Integer16<T>)
            return V256<TO> {v};
         else if constexpr (CT::Integer32<T>)
            return V256<TO> {v.Pack()};
         else if constexpr (CT::Integer64<T>)
            return V256<TO> {v.Pack().Pack()};
         else
            static_assert(false, "Unsupported conversion");
      }
      else if constexpr (CT::Integer32<TO>) {
         //                                                             
         // Converting to 32bit integer                                 
         //                                                             
         if constexpr (CT::Integer8<T>)
            return V256<TO> {v.UnpackLo().UnpackLo()};
         else if constexpr (CT::Integer16<T>)
            return V256<TO> {v.UnpackLo()};
         else if constexpr (CT::Integer32<T>)
            return V256<TO> {v};
         else if constexpr (CT::Integer64<T>)
            return V256<TO> {v.Pack()};
         else
            static_assert(false, "Unsupported conversion");
      }
      else if constexpr (CT::Integer64<TO>) {
         //                                                             
         // Converting to 64bit integer                                 
         //                                                             
         if constexpr (CT::Integer8<T>)
            return V256<TO> {v.UnpackLo().UnpackLo().UnpackLo()};
         else if constexpr (CT::Integer16<T>)
            return V256<TO> {v.UnpackLo().UnpackLo()};
         else if constexpr (CT::Integer32<T>)
            return V256<TO> {v.UnpackLo()};
         else if constexpr (CT::Integer64<T>)
            return V256<TO> {v};
         else
            static_assert(false, "Unsupported conversion");
      }
      else static_assert(false, "Unsupported register");
   }

} // namespace Langulus::SIMD