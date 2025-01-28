///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "../Common.hpp"
#include "From128d.hpp"


namespace Langulus::SIMD::Inner
{

   /// https://stackoverflow.com/questions/41144668                           
   /// Converts the lower two floats into uint64                              
   ///   @attention only works for inputs in the range: [0, 2^52)             
   LANGULUS(INLINED)
   simde__m128i float_to_uint64(simde__m128 x) noexcept {
      return double_to_uint64(simde_mm_cvtps_pd(x));
   }

   /// Converts the lower two floats into int64                               
   ///   @attention only works for inputs in the range: [-2^51, 2^51]         
   LANGULUS(INLINED)
   simde__m128i float_to_int64(simde__m128d x) noexcept {
      return double_to_int64(simde_mm_cvtps_pd(x));
   }


   /// Convert V128f to any other register                                    
   ///   @tparam TO - the desired element type                                
   ///   @param v - the input register                                        
   ///   @return the converted register                                       
   template<Element TO> LANGULUS(INLINED)
   auto ConvertFrom128f(CT::SIMD128f auto v) noexcept {
      if constexpr (CT::Double<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 32bit floats -> 64bit floats");
         #if LANGULUS_SIMD(AVX)
            return V256<TO> {simde_mm256_cvtps_pd(v)};
         #else
            return V128<TO> {simde_mm_cvtps_pd(v)};
         #endif
      }
      else if constexpr (CT::Float<TO>) {
         LANGULUS_SIMD_VERBOSE("No conversion required");
         return v;
      }
      else if constexpr (CT::SignedInteger8<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 32bit floats -> signed 8bit integers");
         const V128i32 t32 {simde_mm_cvtps_epi32(v)};
         return t32.Pack().Pack();
      }
      else if constexpr (CT::UnsignedInteger8<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 32bit floats -> unsigned 8bit integers");
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            const V128u32 t32 {simde_mm_cvtps_epu32(v)};
         #else
            const V128u32 t32 {simde_mm_cvtps_epi32(v)};
         #endif
         return t32.Pack().Pack();
      }
      else if constexpr (CT::SignedInteger16<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 32bit floats -> signed 16bit integers");
         const V128i32 t32 {simde_mm_cvtps_epi32(v)};
         return t32.Pack();
      }
      else if constexpr (CT::UnsignedInteger16<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 32bit floats -> unsigned 16bit integers");
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            const V128u32 t32 {simde_mm_cvtps_epu32(v)};
         #else
            const V128u32 t32 {simde_mm_cvtps_epi32(v)};
         #endif
         return t32.Pack();
      }
      else if constexpr (CT::SignedInteger32<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 32bit floats -> signed 32bit integers");
         return V128<TO> {simde_mm_cvtps_epi32(v)};
      }
      else if constexpr (CT::UnsignedInteger32<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 32bit floats -> unsigned 32bit integers");
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            return V128<TO> {simde_mm_cvtps_epu32(v)};
         #else
            return V128<TO> {simde_mm_cvtps_epi32(v)};
         #endif
      }
      else if constexpr (CT::SignedInteger64<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 32bit floats -> signed 64bit integers");
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return V128<TO> {simde_mm_cvtps_epi64(v)};
         #elif LANGULUS_SIMD(AVX)
            const V256i32 t32 {simde_mm256_cvtps_epi32(simde_mm256_castps128_ps256(v))};
            return t32.UnpackLo();
         #else
            return V128<TO> {float_to_int64(v)};
         #endif
      }
      else if constexpr (CT::UnsignedInteger64<TO>) {
         LANGULUS_SIMD_VERBOSE("Converting 32bit floats -> unsigned 64bit integers");
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return V128<TO> {simde_mm_cvtps_epu64(v)};
         #elif LANGULUS_SIMD(AVX)
            const V256u32 t32 {simde_mm256_cvtps_epi32(simde_mm256_castps128_ps256(v))};
            return t32.UnpackLo();
         #else
            return V128<TO> {float_to_uint64(v)};
         #endif
      }
      else static_assert(false, "Unsupported register");
   }

} // namespace Langulus::SIMD