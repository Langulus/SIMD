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
      constexpr Unsupported AddSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Add two registers                                                   
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      R AddSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;

         if constexpr (CT::SIMD128<R>) {
            if      constexpr (CT::SignedInteger8<T>)    return simde_mm_add_epi8      (lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)  return simde_mm_adds_epu8     (lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)   return simde_mm_add_epi16     (lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>) return simde_mm_adds_epu16    (lhs, rhs);
            else if constexpr (CT::Integer32<T>)         return simde_mm_add_epi32     (lhs, rhs);
            else if constexpr (CT::Integer64<T>)         return simde_mm_add_epi64     (lhs, rhs);
            else if constexpr (CT::Float<T>)             return simde_mm_add_ps        (lhs, rhs);
            else if constexpr (CT::Double<T>)            return simde_mm_add_pd        (lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if      constexpr (CT::SignedInteger8<T>)    return simde_mm256_add_epi8   (lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)  return simde_mm256_adds_epu8  (lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)   return simde_mm256_add_epi16  (lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>) return simde_mm256_adds_epu16 (lhs, rhs);
            else if constexpr (CT::Integer32<T>)         return simde_mm256_add_epi32  (lhs, rhs);
            else if constexpr (CT::Integer64<T>)         return simde_mm256_add_epi64  (lhs, rhs);
            else if constexpr (CT::Float<T>)             return simde_mm256_add_ps     (lhs, rhs);
            else if constexpr (CT::Double<T>)            return simde_mm256_add_pd     (lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if      constexpr (CT::SignedInteger8<T>)    return simde_mm512_add_epi8   (lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)  return simde_mm512_adds_epu8  (lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)   return simde_mm512_add_epi16  (lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>) return simde_mm512_adds_epu16 (lhs, rhs);
            else if constexpr (CT::Integer32<T>)         return simde_mm512_add_epi32  (lhs, rhs);
            else if constexpr (CT::Integer64<T>)         return simde_mm512_add_epi64  (lhs, rhs);
            else if constexpr (CT::Float<T>)             return simde_mm512_add_ps     (lhs, rhs);
            else if constexpr (CT::Double<T>)            return simde_mm512_add_pd     (lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }
      
      /// Get sum of values as constexpr, if possible                         
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the summed scalar/vector                                  
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto AddConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> E {
               return l + r;
            }
         );
      }
   
      /// Get summed values as a register, if possible                        
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the summed scalar/vector/register                         
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Add(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Adding (SIMD) as ", NameOf<R>());
               return AddSIMD(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> E {
               LANGULUS_SIMD_VERBOSE("Adding (Fallback) ", l, " + ", r, " (", NameOf<E>(), ")");
               return l + r;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(Add)

} // namespace Langulus::SIMD
