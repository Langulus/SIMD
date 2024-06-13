///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "../Attempt.hpp"
#include "../Convert.hpp"
#include "Equals.hpp"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      NOD() LANGULUS(INLINED)
      constexpr Unsupported DivideSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Divide two registers                                                
      ///   @attention will throw if any element on right side is zero        
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      R DivideSIMD(R lhs, R rhs) {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;

         if constexpr (CT::SIMD128<R>) {
            // Check if anything in 'rhs' is zero                       
            if constexpr (CT::Integer<T>) {
               if (simde_mm_movemask_epi8(EqualsSIMD(rhs, rhs.Zero())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
            }
            else if constexpr (CT::Float<T>) {
               if (simde_mm_movemask_ps(EqualsSIMD(rhs, rhs.Zero())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
            }
            else if constexpr (CT::Double<T>) {
               if (simde_mm_movemask_pd(EqualsSIMD(rhs, rhs.Zero())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
            }
            else LANGULUS_ERROR("Unsupported T");

            // Divide                                                   
            if constexpr (CT::UnsignedInteger8<T>)       return simde_mm_div_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger8<T>)    return simde_mm_div_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>) return simde_mm_div_epu16(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)   return simde_mm_div_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>) return simde_mm_div_epu32(lhs, rhs);
            else if constexpr (CT::SignedInteger32<T>)   return simde_mm_div_epi32(lhs, rhs);
            else if constexpr (CT::UnsignedInteger64<T>) return simde_mm_div_epu64(lhs, rhs);
            else if constexpr (CT::SignedInteger64<T>)   return simde_mm_div_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)             return simde_mm_div_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)            return simde_mm_div_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            // Check if anything in 'rhs' is zero                       
            if constexpr (CT::Integer<T>) {
               if (simde_mm256_movemask_epi8(EqualsSIMD(rhs, rhs.Zero())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
            }
            else if constexpr (CT::Float<T>) {
               if (simde_mm256_movemask_ps(EqualsSIMD(rhs, rhs.Zero())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
            }
            else if constexpr (CT::Double<T>) {
               if (simde_mm256_movemask_pd(EqualsSIMD(rhs, rhs.Zero())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
            }
            else LANGULUS_ERROR("Unsupported T");

            // Divide                                                   
            if constexpr (CT::UnsignedInteger8<T>)       return simde_mm256_div_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger8<T>)    return simde_mm256_div_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>) return simde_mm256_div_epu16(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)   return simde_mm256_div_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>) return simde_mm256_div_epu32(lhs, rhs);
            else if constexpr (CT::SignedInteger32<T>)   return simde_mm256_div_epi32(lhs, rhs);
            else if constexpr (CT::UnsignedInteger64<T>) return simde_mm256_div_epu64(lhs, rhs);
            else if constexpr (CT::SignedInteger64<T>)   return simde_mm256_div_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)             return simde_mm256_div_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)            return simde_mm256_div_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            // Check if anything in 'rhs' is zero                       
            if constexpr (CT::Integer<T>) {
               if (EqualsSIMD(rhs, rhs.Zero()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
            }
            else if constexpr (CT::Float<T>) {
               if (EqualsSIMD(rhs, rhs.Zero()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
            }
            else if constexpr (CT::Double<T>) {
               if (EqualsSIMD(rhs, rhs.Zero()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
            }
            else LANGULUS_ERROR("Unsupported T");

            // Divide                                                   
            if constexpr (CT::UnsignedInteger8<T>)       return simde_mm512_div_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger8<T>)    return simde_mm512_div_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>) return simde_mm512_div_epu16(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)   return simde_mm512_div_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>) return simde_mm512_div_epu32(lhs, rhs);
            else if constexpr (CT::SignedInteger32<T>)   return simde_mm512_div_epi32(lhs, rhs);
            else if constexpr (CT::UnsignedInteger64<T>) return simde_mm512_div_epu64(lhs, rhs);
            else if constexpr (CT::SignedInteger64<T>)   return simde_mm512_div_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)             return simde_mm512_div_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)            return simde_mm512_div_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }

      
      /// Get divided values as constexpr, if possible                        
      ///   @attention will throw on division by zero                         
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the divided scalar/vector                                 
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto DivideConstexpr(const auto& lhs, const auto& rhs) {
         return AttemptBinary<1, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) -> E {
               if (r == E {0})
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return l / r;
            }
         );
      }
   
      /// Get divided values as a register, if possible                       
      ///   @attention will throw on division by zero                         
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the divided scalar/vector/register                        
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Divide(const auto& lhs, const auto& rhs) {
         return AttemptBinary<1, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) {
               LANGULUS_SIMD_VERBOSE("Dividing (SIMD) as ", NameOf<R>());
               return DivideSIMD(l, r);
            },
            []<class E>(const E& l, const E& r) -> E {
               LANGULUS_SIMD_VERBOSE("Dividing (Fallback) ", l, " / ", r, " (", NameOf<E>(), ")");
               if (r == E {0})
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return l / r;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner


   /// Divide numbers, and force output to desired place                      
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in 'out'. Use Inner::Divide if you     
   ///      don't want this.                                                  
   template<class LHS, class RHS, CT::NotSemantic OUT> LANGULUS(INLINED)
   constexpr void Divide(const LHS& lhs, const RHS& rhs, OUT& out) {
      if constexpr (CT::SIMD<OUT>)
         out = Inner::Divide<OUT>(lhs, rhs);
      else if constexpr (CT::SIMD<LHS> or CT::SIMD<RHS>)
         Store(Inner::Divide<OUT>(lhs, rhs), out);
      else {
         IF_CONSTEXPR() {
            Store(Inner::DivideConstexpr<OUT>(DesemCast(lhs), DesemCast(rhs)), out);
         }
         else {
            Store(Inner::Divide<OUT>(DesemCast(lhs), DesemCast(rhs)), out);
         }
      }
   }

   /// Divide numbers                                                         
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in an instance of 'OUT'. Use           
   ///      Inner::Divide if you don't want this.                             
   template<class LHS, class RHS, CT::NotSemantic OUT = LosslessArray<LHS, RHS>>
   LANGULUS(INLINED)
   constexpr auto Divide(const LHS& lhs, const RHS& rhs) {
      OUT out;
      Divide(DesemCast(lhs), DesemCast(rhs), out);

      if constexpr (CT::Similar<LHS, RHS> or CT::DerivedFrom<LHS, RHS>)
         return LHS {out};
      else if constexpr (CT::DerivedFrom<RHS, LHS>)
         return RHS {out};
      else
         return out;
   }

} // namespace Langulus::SIMD
