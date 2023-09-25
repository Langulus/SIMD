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

      /// Get floored values via SIMD                                         
      ///   @tparam T - the type of the array element                         
      ///   @tparam S - the size of the array                                 
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param value - the array                                          
      ///   @return the floored values                                        
      template<class T, Count S, CT::SIMD REGISTER>
      LANGULUS(INLINED)
      auto Floor(const REGISTER& value) noexcept {
         static_assert(CT::Real<T>, "Suboptimal and pointless for whole numbers");

         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm_floor_ps(value);
            else if constexpr (CT::Double<T>)
               return simde_mm_floor_pd(value);
            else
               LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm256_floor_ps(value);
            else if constexpr (CT::Double<T>)
               return simde_mm256_floor_pd(value);
            else
               LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm512_floor_ps(value);
            else if constexpr (CT::Double<T>)
               return simde_mm512_floor_pd(value);
            else
               LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Get the floor values                                                   
   ///   @param T - type of a single value                                    
   ///   @param S - size of the array                                         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<class T, Count S>
   LANGULUS(INLINED)
   auto Floor(const T(&value)[S]) noexcept {
      return Inner::Floor<T, S>(Load<0>(value));
   }

} // namespace Langulus::SIMD