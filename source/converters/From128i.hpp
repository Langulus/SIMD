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

   /// Convert V128i to any other register                                    
   ///   @tparam TO - the desired element type                                
   ///   @param v - the input register                                        
   ///   @return the converted register                                       
   template<Element TO> NOD() LANGULUS(INLINED)
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