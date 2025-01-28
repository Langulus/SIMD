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
   /// Converts the lower two floats into uint64                              
   ///   @attention only works for inputs in the range: [0, 2^52)             
   LANGULUS(INLINED)
   simde__m128i double_to_uint64(simde__m128d x) noexcept {
      x = simde_mm_add_pd(x, simde_mm_set1_pd(0x0010000000000000));
      return simde_mm_xor_si128(
         simde_mm_castpd_si128(x),
         simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000))
      );
   }

   /// Converts the lower two floats into int64                               
   ///   @attention only works for inputs in the range: [-2^51, 2^51]         
   LANGULUS(INLINED)
   simde__m128i double_to_int64(simde__m128d x) noexcept {
      x = simde_mm_add_pd(x, simde_mm_set1_pd(0x0018000000000000));
      return simde_mm_sub_epi64(
         simde_mm_castpd_si128(x),
         simde_mm_castpd_si128(simde_mm_set1_pd(0x0018000000000000))
      );
   }

   /// Convert V128d to any other register                                    
   ///   @tparam TO - the desired element type                                
   ///   @param v - the input register                                        
   ///   @return the converted register                                       
   template<Element TO> LANGULUS(INLINED)
   auto ConvertFrom128d(CT::SIMD128d auto v) noexcept {
      if constexpr (CT::Double<TO>) {
         LANGULUS_SIMD_VERBOSE("No conversion required");
         return v;
      }
      else if constexpr (CT::Float<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> 32bit floats");
         return V128<TO> {simde_mm_cvtpd_ps(v)};
      }
      else if constexpr (CT::SignedInteger8<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> signed 8bit integers");
         const V128i32 t32 {simde_mm_cvtpd_epi32(v)};
         return t32.Pack().Pack();
      }
      else if constexpr (CT::UnsignedInteger8<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> unsigned 8bit integers");
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            const V128u32 t32 {simde_mm_cvtpd_epu32(v)};
         #else
            const V128u32 t32 {simde_mm_cvtpd_epi32(v)};
         #endif
         return t32.Pack().Pack();
      }
      else if constexpr (CT::SignedInteger16<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> signed 16bit integers");
         const V128i32 t32 {simde_mm_cvtpd_epi32(v)};
         return t32.Pack();
      }
      else if constexpr (CT::UnsignedInteger16<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> unsigned 16bit integers");
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            const V128u32 t32 {simde_mm_cvtpd_epu32(v)};
         #else
            const V128u32 t32 {simde_mm_cvtpd_epi32(v)};
         #endif
         return t32.Pack();
      }
      else if constexpr (CT::SignedInteger32<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> signed 32bit integers");
         return V128<TO> {simde_mm_cvtpd_epi32(v)};
      }
      else if constexpr (CT::UnsignedInteger32<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> unsigned 32bit integers");
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            return V128<TO> {simde_mm_cvtpd_epu32(v)};
         #else
            return V128<TO> {simde_mm_cvtpd_epi32(v)};
         #endif
      }
      else if constexpr (CT::SignedInteger64<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> signed 64bit integers");
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return V128<TO> {simde_mm_cvtpd_epi64(v)};
         #else
            return V128<TO> {double_to_int64(v)};
         #endif
      }
      else if constexpr (CT::UnsignedInteger64<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> unsigned 64bit integers");
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return V128<TO> {simde_mm_cvtpd_epu64(v)};
         #else
            return V128<TO> {double_to_uint64(v)};
         #endif
      }
      else static_assert(false, "Unsupported register");
   }

} // namespace Langulus::SIMD