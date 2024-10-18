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

   /// Convert V256f to any other register                                    
   ///   @tparam TO - the desired element type                                
   ///   @param v - the input register                                        
   ///   @return the converted register                                       
   template<Element TO> NOD() LANGULUS(INLINED)
   auto ConvertFrom256f(CT::SIMD256f auto v) noexcept {
      if constexpr (CT::Double<TO>)
         return V256<TO> {simde_mm256_cvtps_pd(simde_mm256_castps256_ps128(v))};
      else if constexpr (CT::Float<TO>)
         return v;
      else if constexpr (CT::SignedInteger8<TO>) {
         const V256i32 t32 {simde_mm256_cvtps_epi32(v)};
         return t32.Pack().Pack();
      }
      else if constexpr (CT::UnsignedInteger8<TO>) {
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            const V256u32 t32 {simde_mm256_cvtps_epu32(v)};
         #else
            const V256u32 t32 {simde_mm256_cvtps_epi32(v)};
         #endif
         return t32.Pack().Pack();
      }
      else if constexpr (CT::SignedInteger16<TO>) {
         const V256i32 t32 {simde_mm256_cvtps_epi32(v)};
         return t32.Pack();
      }
      else if constexpr (CT::UnsignedInteger16<TO>) {
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            const V256u32 t32 {simde_mm256_cvtps_epu32(v)};
         #else
            const V256u32 t32 {simde_mm256_cvtps_epi32(v)};
         #endif
         return t32.Pack();
      }
      else if constexpr (CT::SignedInteger32<TO>)
         return V256<TO> {simde_mm256_cvtps_epi32(v)};
      else if constexpr (CT::UnsignedInteger32<TO>) {
         #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
            return V256<TO> {simde_mm256_cvtps_epu32(v)};
         #else
            return V256<TO> {simde_mm256_cvtps_epi32(v)};
         #endif
      }
      else if constexpr (CT::SignedInteger64<TO>) {
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return V256<TO> {simde_mm256_cvtps_epi64(v)};
         #else
            const V256i32 t32 {simde_mm256_cvtps_epi32(v)};
            return t32.UnpackLo();
         #endif
      }
      else if constexpr (CT::UnsignedInteger64<TO>) {
         #if LANGULUS_SIMD(AVX512DQ) and LANGULUS_SIMD(AVX512VL)
            return V256<TO> {simde_mm256_cvtps_epu64(v)};
         #else
            const V256u32 t32 {simde_mm256_cvtps_epi32(v)};
            return t32.UnpackLo();
         #endif
      }
      else static_assert(false, "Unsupported register");
   }

} // namespace Langulus::SIMD