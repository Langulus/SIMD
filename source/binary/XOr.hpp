///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "../Attempt.hpp"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      NOD() LANGULUS(INLINED)
      constexpr Unsupported XOrSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// XOr using registers                                                 
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      R XOrSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;

         if constexpr (CT::SIMD128<R>) {
            if      constexpr (CT::Integer<T>)  return simde_mm_xor_si128   (lhs, rhs);
            else if constexpr (CT::Float<T>)    return simde_mm_xor_ps      (lhs, rhs);
            else if constexpr (CT::Double<T>)   return simde_mm_xor_pd      (lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if      constexpr (CT::Integer<T>)  return simde_mm256_xor_si256(lhs, rhs);
            else if constexpr (CT::Float<T>)    return simde_mm256_xor_ps   (lhs, rhs);
            else if constexpr (CT::Double<T>)   return simde_mm256_xor_pd   (lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if      constexpr (CT::Integer<T>)  return simde_mm512_xor_si512(lhs, rhs);
            else if constexpr (CT::Float<T>)    return simde_mm512_xor_ps   (lhs, rhs);
            else if constexpr (CT::Double<T>)   return simde_mm512_xor_pd   (lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }
      
      /// Xor values as constexpr, if possible                                
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the scalar/vector                                         
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto XOrConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> E {
               return l ^ r;
            }
         );
      }
   
      /// Xor values as a register, if possible                               
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the scalar/vector/register                                
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto XOr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Xoring (SIMD) as ", NameOf<R>());
               return XOrSIMD(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> E {
               LANGULUS_SIMD_VERBOSE("Xoring (Fallback) ", l, " ^ ", r, " (", NameOf<E>(), ")");
               return l ^ r;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(XOr)

} // namespace Langulus::SIMD