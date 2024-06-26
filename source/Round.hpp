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
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param value - the array                                          
      ///   @return the floored values                                        
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto Round(UNUSED() const REGISTER& value) noexcept {
         static_assert(CT::Real<T>, "Suboptimal for unreal numbers");

      #if LANGULUS_SIMD(128BIT)
         constexpr auto STYLE = SIMDE_MM_FROUND_TO_NEAREST_INT | SIMDE_MM_FROUND_NO_EXC;

         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm_round_ps(value, STYLE);
            else if constexpr (CT::Double<T>)
               return simde_mm_round_pd(value, STYLE);
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm256_round_ps(value, STYLE);
            else if constexpr (CT::Double<T>)
               return simde_mm256_round_pd(value, STYLE);
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm512_roundscale_ps(value, STYLE);
            else if constexpr (CT::Double<T>)
               return simde_mm512_roundscale_pd(value, STYLE);
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else
      #endif

         LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Get the rounded values                                                 
   ///   @param T - type of a single value                                    
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<CT::NotSemantic T> LANGULUS(INLINED)
   auto Round(const T& value) noexcept {
      using DT = Decay<TypeOf<T>>;
      return Inner::Round<DT>(Load<0>(value));
   }

} // namespace Langulus::SIMD