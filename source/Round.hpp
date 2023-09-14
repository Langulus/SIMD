///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Fill.hpp"
#include "Convert.hpp"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Get rounded values via SIMD                                         
      ///   @tparam T - the type of the array element                         
      ///   @tparam S - the size of the array                                 
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param value - the array                                          
      ///   @return the floored values                                        
      template<class T, Count S, CT::TSIMD REGISTER>
      LANGULUS(INLINED)
      auto Round(const REGISTER& value) noexcept {
         static_assert(CT::Real<T>, "Suboptimal for unreal numbers");

         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm_round_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            else if constexpr (CT::Double<T>)
               return simde_mm_round_pd(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm256_round_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            else if constexpr (CT::Double<T>)
               return simde_mm256_round_pd(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm512_roundscale_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            else if constexpr (CT::Double<T>)
               return simde_mm512_roundscale_pd(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Get the rounded values                                                 
   ///   @param T - type of a single value                                    
   ///   @param S - size of the array                                         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<class T, Count S>
   LANGULUS(INLINED)
   auto Round(const T(&value)[S]) noexcept {
      return Inner::Round<T, S>(Load<0>(value));
   }

} // namespace Langulus::SIMD