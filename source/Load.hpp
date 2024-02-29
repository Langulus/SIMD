///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "SetGet.hpp"
#include "IgnoreWarningsPush.inl"


namespace Langulus::SIMD
{

   /// Wrap an array into a register                                          
   ///   @tparam DEF - default number for setting elements outside S          
   ///   @tparam FROM - the array/vector to load (deducible)                  
   ///   @param v - the array/vector to load inside a register                
   ///   @return the register if loaded, or Unsupported if not                
   template<int DEF, class FROM> LANGULUS(INLINED)
   auto Load(UNUSED() const FROM& v) noexcept {
      UNUSED() constexpr auto S = CountOf<FROM>;
      if constexpr (S < 2) {
         // Scalars not allowed                                         
         return Unsupported {};
      }
      else {
         // Vectors are allowed                                         
         using T = TypeOf<FROM>;
         UNUSED() constexpr auto denseSize = sizeof(Decay<T>) * S;

         #if LANGULUS_SIMD(128BIT)
            if constexpr (denseSize <= 16) {
               LANGULUS_SIMD_VERBOSE(
                  "Loading 128bit register from ", S, " unaligned elements");

               // Load as a single 128bit register                      
               if constexpr (denseSize == 16 and CT::Dense<T>) {
                  if constexpr (CT::Float<T>)
                     return simde_mm_loadu_ps(&Inner::GetFirst(v));
                  else if constexpr (CT::Double<T>)
                     return simde_mm_loadu_pd(&Inner::GetFirst(v));
                  else
                     return simde_mm_loadu_si128(&v);
               }
               else return Set<DEF, 16>(v);
            }
            else
         #endif

         #if LANGULUS_SIMD(256BIT)
            if constexpr (denseSize <= 32) {
               LANGULUS_SIMD_VERBOSE(
                  "Loading 256bit register from ", S, " unaligned elements");

               // Load as a single 256bit register                      
               if constexpr (denseSize == 32 and CT::Dense<T>) {
                  if constexpr (CT::Float<T>)
                     return simde_mm256_loadu_ps(&Inner::GetFirst(v));
                  else if constexpr (CT::Double<T>)
                     return simde_mm256_loadu_pd(&Inner::GetFirst(v));
                  else
                     return simde_mm256_loadu_si256(&v);
               }
               else return Set<DEF, 32>(v);
            }
            else
         #endif

         #if LANGULUS_SIMD(512BIT)
            if constexpr (denseSize <= 64) {
               LANGULUS_SIMD_VERBOSE(
                  "Loading 512bit register from ", S, " unaligned elements");

               // Load as a single 512bit register                      
               if constexpr (denseSize == 64 and CT::Dense<T>) {
                  if constexpr (CT::Float<T>)
                     return simde_mm512_loadu_ps(&Inner::GetFirst(v));
                  else if constexpr (CT::Double<T>)
                     return simde_mm512_loadu_pd(&Inner::GetFirst(v));
                  else
                     return simde_mm512_loadu_si512(&v);
               }
               else return Set<DEF, 64>(v);
            }
            else
         #endif

         return Unsupported {};
      }
   }

   namespace Inner
   {

      /// Pick an array that can represent elements of FROM as type TO        
      template<class FROM, class TO = FROM>
      using ToArray = Decay<TypeOf<TO>> [CountOf<FROM>];

      /// Pick a register that can wrap an array/vector/scalar                
      template<class FROM, class TO = FROM>
      using ToSIMD = decltype(Load<0>(Fake<ToArray<FROM, TO>>()));

      /// Determine a SIMD register type that can best wrap LHS and RHS       
      template<class LHS, class RHS, class OUT>
      using Register = Conditional<
         ((CountOf<LHS>) > (CountOf<RHS>)),
         ToSIMD<LHS, OUT>, ToSIMD<RHS, OUT>
      >;

   } // namespace Langulus::SIMD::Inner

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
