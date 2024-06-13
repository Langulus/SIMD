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
      constexpr Unsupported SubtractSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Subtract two registers                                              
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      R SubtractSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;
         
         if constexpr (CT::SIMD128<R>) {
            if      constexpr (CT::Integer8<T>)    return simde_mm_sub_epi8    (lhs, rhs);
            else if constexpr (CT::Integer16<T>)   return simde_mm_sub_epi16   (lhs, rhs);
            else if constexpr (CT::Integer32<T>)   return simde_mm_sub_epi32   (lhs, rhs);
            else if constexpr (CT::Integer64<T>)   return simde_mm_sub_epi64   (lhs, rhs);
            else if constexpr (CT::Float<T>)       return simde_mm_sub_ps      (lhs, rhs);
            else if constexpr (CT::Double<T>)      return simde_mm_sub_pd      (lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if      constexpr (CT::Integer8<T>)    return simde_mm256_sub_epi8 (lhs, rhs);
            else if constexpr (CT::Integer16<T>)   return simde_mm256_sub_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)   return simde_mm256_sub_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)   return simde_mm256_sub_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)       return simde_mm256_sub_ps   (lhs, rhs);
            else if constexpr (CT::Double<T>)      return simde_mm256_sub_pd   (lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if      constexpr (CT::Integer8<T>)    return simde_mm512_sub_epi8 (lhs, rhs);
            else if constexpr (CT::Integer16<T>)   return simde_mm512_sub_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)   return simde_mm512_sub_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)   return simde_mm512_sub_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)       return simde_mm512_sub_ps   (lhs, rhs);
            else if constexpr (CT::Double<T>)      return simde_mm512_sub_pd   (lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }
      
      /// Get difference of values as constexpr, if possible                  
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the difference scalar/vector                              
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto SubtractConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> E {
               return l - r;
            }
         );
      }
   
      /// Get difference of values as a register, if possible                 
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the difference scalar/vector/register                     
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Subtract(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Subtracting (SIMD) as ", NameOf<R>());
               return SubtractSIMD(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> E {
               LANGULUS_SIMD_VERBOSE("Subtracting (Fallback) ", l, " - ", r, " (", NameOf<E>(), ")");
               return l - r;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(Subtract)

} // namespace Langulus::SIMD