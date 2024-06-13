///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "Common.hpp"


namespace Langulus::SIMD
{

   /// Fill a register with a single value                                    
   ///   @tparam R - register size                                            
   ///   @param s -  the scalar value to use for filling                      
   ///   @return the filled register                                          
   template<int R> NOD() LANGULUS(INLINED)
   auto Fill(const CT::Scalar auto& s) noexcept {
      using T = Deref<decltype(s)>;

      #if LANGULUS_SIMD(128BIT)
         if constexpr (R <= 16) {
            if      constexpr (CT::SignedInteger8<T>)    return V128<T> {simde_mm_set1_epi8        ( GetFirst(s))};
            else if constexpr (CT::UnsignedInteger8<T>)  return V128<T> {simde_x_mm_set1_epu8      ( GetFirst(s))};
            else if constexpr (CT::SignedInteger16<T>)   return V128<T> {simde_mm_set1_epi16       ( GetFirst(s))};
            else if constexpr (CT::UnsignedInteger16<T>) return V128<T> {simde_x_mm_set1_epu16     ( GetFirst(s))};
            else if constexpr (CT::SignedInteger32<T>)   return V128<T> {simde_mm_set1_epi32       ( GetFirst(s))};
            else if constexpr (CT::UnsignedInteger32<T>) return V128<T> {simde_x_mm_set1_epu32     ( GetFirst(s))};
            else if constexpr (CT::SignedInteger64<T>)   return V128<T> {simde_mm_set1_epi64x      ( GetFirst(s))};
            else if constexpr (CT::UnsignedInteger64<T>) return V128<T> {simde_x_mm_set1_epu64     ( GetFirst(s))};
            else if constexpr (CT::Float<T>)             return V128<T> {simde_mm_broadcast_ss     (&GetFirst(s))};
            else if constexpr (CT::Double<T>)            return V128<T> {simde_mm_set1_pd          ( GetFirst(s))};
            else LANGULUS_ERROR("Unsupported type for filling V128");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (R <= 32) {
            if      constexpr (CT::Integer8<T>)          return V256<T> {simde_mm256_set1_epi8     ( GetFirst(s))};
            else if constexpr (CT::Integer16<T>)         return V256<T> {simde_mm256_set1_epi16    ( GetFirst(s))};
            else if constexpr (CT::Integer32<T>)         return V256<T> {simde_mm256_set1_epi32    ( GetFirst(s))};
            else if constexpr (CT::Integer64<T>)         return V256<T> {simde_mm256_set1_epi64x   ( GetFirst(s))};
            else if constexpr (CT::Float<T>)             return V256<T> {simde_mm256_broadcast_ss  (&GetFirst(s))};
            else if constexpr (CT::Double<T>)            return V256<T> {simde_mm256_broadcast_sd  (&GetFirst(s))};
            else LANGULUS_ERROR("Unsupported type for filling of V256");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (R <= 64) {
            if      constexpr (CT::Integer8<T>)          return V512<T> {simde_mm512_set1_epi8     (GetFirst(s))};
            else if constexpr (CT::Integer16<T>)         return V512<T> {simde_mm512_set1_epi16    (GetFirst(s))};
            else if constexpr (CT::Integer32<T>)         return V512<T> {simde_mm512_set1_epi32    (GetFirst(s))};
            else if constexpr (CT::Integer64<T>)         return V512<T> {simde_mm512_set1_epi64    (GetFirst(s))};
            else if constexpr (CT::Float<T>)             return V512<T> {simde_mm512_set1_ps       (GetFirst(s))};
            else if constexpr (CT::Double<T>)            return V512<T> {simde_mm512_set1_pd       (GetFirst(s))};
            else LANGULUS_ERROR("Unsupported type for filling of V512");
         }
         else
      #endif
      
      return Unsupported {};
   }

} // namespace Langulus::SIMD
