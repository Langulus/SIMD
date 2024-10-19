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

   /// Convert V128d to any other register                                    
   ///   @tparam TO - the desired element type                                
   ///   @param v - the input register                                        
   ///   @return the converted register                                       
   template<Element TO> NOD() LANGULUS(INLINED)
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
            const V128i32 t32 {simde_mm_cvtpd_epi32(v)};
            return t32.UnpackLo();
         #endif
      }
      else if constexpr (CT::UnsignedInteger64<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 64bit floats -> unsigned 64bit integers");
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return V128<TO> {simde_mm_cvtpd_epu64(v)};
         #else
            const V128u32 t32 {simde_mm_cvtpd_epi32(v)};
            return t32.UnpackLo();
         #endif
      }
      else static_assert(false, "Unsupported register");
   }

} // namespace Langulus::SIMD