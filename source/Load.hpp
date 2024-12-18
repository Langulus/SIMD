///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "SetGet.hpp"
#include "Fill.hpp"


namespace Langulus::SIMD
{
   namespace Inner
   {
      template<class R, class FORCE_OUT>
      consteval Count DecideCount() {
         using T = Decvq<TypeOf<R>>;
         if constexpr (CT::Void<FORCE_OUT>)
            return CountOf<R>;
         else 
            return sizeof(FORCE_OUT) / sizeof(T);
      }
   }

   /// Load a register into another register                                  
   ///   @tparam DEF - default value for setting elements outside input size  
   ///   @tparam FORCE_OUT - the type of register we want as a result - use   
   ///      void to pick a register that is capable of fitting 'v'            
   ///   @param v - the register to load                                      
   ///   @return the register, or Unsupported if not possible                 
   template<auto DEF, class FORCE_OUT = void> NOD() LANGULUS(INLINED)
   auto Load(const CT::SIMD auto& v) noexcept {
      using R = Deref<decltype(v)>;
      constexpr auto S = Inner::DecideCount<R, FORCE_OUT>();

      if constexpr (S == CountOf<R>) {
         // Just forward the original register                          
         return v;
      }
      else {
         // Cast into another type of regiter, filling the blanks with  
         // DEF value                                                   
         static_assert(false, "TODO use _mm_cast...");
      }
   }

   /// Wrap an array/scalar into a register                                   
   ///   @tparam DEF - default value for setting elements outside input size  
   ///   @tparam FORCE_OUT - the type of register we want as a result - use   
   ///      void to pick a register that is capable of fitting 'v'            
   ///   @param v - the scalar/vector to load inside a register               
   ///   @return the register, or Unsupported if not possible                 
   template<auto DEF, class FORCE_OUT = void> NOD() LANGULUS(INLINED)
   auto Load(const CT::NotSIMD auto& v) noexcept {
      using R = Deref<decltype(v)>;
      using T = Decvq<TypeOf<R>>;

      if constexpr (CT::Scalar<R>) {
         if constexpr (CT::Void<FORCE_OUT> or CountOf<FORCE_OUT> == 1) {
            // We either can't decide in what register to insert the    
            // scalar, or the destination is just a scalar - fallback   
            return Unsupported {};
         }
         else {
            // Load a scalar, by duplicating the value for each element 
            // in the register. FORCE_OUT MUST BE SET!                  
            constexpr auto S = Inner::DecideCount<R, FORCE_OUT>();
            return Fill<sizeof(T) * S>(v);
         }
      }
      else {
         // Load a vector either partially, filling the blanks using    
         // DEF value, or directly if vector is of the proper size      
         // Should perform faster if 'v' is aligned properly            
         constexpr auto S  = Inner::DecideCount<R, FORCE_OUT>();
         UNUSED() constexpr auto RS = sizeof(T) * S;

         #if LANGULUS_SIMD(128BIT)
            if constexpr (RS <= 16) {
               LANGULUS_SIMD_VERBOSE(
                  "Loading 128bit register from ", S, " unaligned elements");

               // Load as a single 128bit register                      
               if constexpr (sizeof(R) >= 16) {
                  if      constexpr (CT::Float<T>)    return V128<T> {simde_mm_loadu_ps   (&GetFirst(v))};
                  else if constexpr (CT::Double<T>)   return V128<T> {simde_mm_loadu_pd   (&GetFirst(v))};
                  else if constexpr (CT::Integer<T>)  return V128<T> {simde_mm_loadu_si128(&GetFirst(v))};
                  else static_assert(false, "Unsupported element");
               }
               else return Set<DEF, 16>(v);
            }
            else
         #endif

         #if LANGULUS_SIMD(256BIT)
            if constexpr (RS <= 32) {
               LANGULUS_SIMD_VERBOSE(
                  "Loading 256bit register from ", S, " unaligned elements");

               // Load as a single 256bit register                      
               if constexpr (sizeof(R) >= 32) {
                  if      constexpr (CT::Float<T>)    return V256<T> {simde_mm256_loadu_ps   (&GetFirst(v))};
                  else if constexpr (CT::Double<T>)   return V256<T> {simde_mm256_loadu_pd   (&GetFirst(v))};
                  else if constexpr (CT::Integer<T>)  return V256<T> {simde_mm256_loadu_si256(&GetFirst(v))};
                  else static_assert(false, "Unsupported element");
               }
               else return Set<DEF, 32>(v);
            }
            else
         #endif

         #if LANGULUS_SIMD(512BIT)
            if constexpr (RS <= 64) {
               LANGULUS_SIMD_VERBOSE(
                  "Loading 512bit register from ", S, " unaligned elements");

               // Load as a single 512bit register                      
               if constexpr (sizeof(R) >= 64) {
                  if      constexpr (CT::Float<T>)    return V512<T> {simde_mm512_loadu_ps   (&GetFirst(v))};
                  else if constexpr (CT::Double<T>)   return V512<T> {simde_mm512_loadu_pd   (&GetFirst(v))};
                  else if constexpr (CT::Integer<T>)  return V512<T> {simde_mm512_loadu_si512(&GetFirst(v))};
                  else static_assert(false, "Unsupported element");
               }
               else return Set<DEF, 64>(v);
            }
            else
         #endif
         return Unsupported {};
      }
   }

} // namespace Langulus::SIMD