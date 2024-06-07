///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "Fill.hpp"
#include "Evaluate.hpp"
#include "IgnoreWarningsPush.inl"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<CT::Decayed, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported DivideSIMD(const T&, const T&) noexcept {
         return {};
      }

      /// Divide two arrays using SIMD                                        
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - type of register we're operating with          
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the divided elements as a register                        
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto DivideSIMD(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) {
      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::UnsignedInteger8<T>) {
               if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi8(rhs, simde_mm_setzero_si128())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_epu8(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger8<T>) {
               if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi8(rhs, simde_mm_setzero_si128())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_epi8(lhs, rhs);
            }
            else if constexpr (CT::UnsignedInteger16<T>) {
               if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi16(rhs, simde_mm_setzero_si128())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_epu16(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger16<T>) {
               if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi16(rhs, simde_mm_setzero_si128())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_epi16(lhs, rhs);
            }
            else if constexpr (CT::UnsignedInteger32<T>) {
               if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi32(rhs, simde_mm_setzero_si128())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_epu32(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger32<T>) {
               if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi32(rhs, simde_mm_setzero_si128())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_epi32(lhs, rhs);
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi64(rhs, simde_mm_setzero_si128())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_epu64(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger64<T>) {
               if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi64(rhs, simde_mm_setzero_si128())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_epi64(lhs, rhs);
            }
            else if constexpr (CT::Float<T>) {
               if (simde_mm_movemask_ps(simde_mm_cmpeq_ps(rhs, simde_mm_setzero_ps())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_ps(lhs, rhs);
            }
            else if constexpr (CT::Double<T>) {
               if (simde_mm_movemask_pd(simde_mm_cmpeq_pd(rhs, simde_mm_setzero_pd())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm_div_pd(lhs, rhs);
            }
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::UnsignedInteger8<T>) {
               if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(rhs, simde_mm256_setzero_si256())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_epu8(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger8<T>) {
               if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(rhs, simde_mm256_setzero_si256())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_epi8(lhs, rhs);
            }
            else if constexpr (CT::UnsignedInteger16<T>) {
               if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi16(rhs, simde_mm256_setzero_si256())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_epu16(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger16<T>) {
               if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi16(rhs, simde_mm256_setzero_si256())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_epi16(lhs, rhs);
            }
            else if constexpr (CT::UnsignedInteger32<T>) {
               if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(rhs, simde_mm256_setzero_si256())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_epu32(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger32<T>) {
               if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(rhs, simde_mm256_setzero_si256())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_epi32(lhs, rhs);
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi64(rhs, simde_mm256_setzero_si256())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_epu64(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger64<T>) {
               if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi64(rhs, simde_mm256_setzero_si256())))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_epi64(lhs, rhs);
            }
            else if constexpr (CT::Float<T>) {
               if (simde_mm256_movemask_ps(simde_mm256_cmp_ps(rhs, simde_mm256_setzero_ps(), _CMP_EQ_OQ)))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_ps(lhs, rhs);
            }
            else if constexpr (CT::Double<T>) {
               if (simde_mm256_movemask_pd(simde_mm256_cmp_pd(rhs, simde_mm256_setzero_pd(), _CMP_EQ_OQ)))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm256_div_pd(lhs, rhs);
            }
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::UnsignedInteger8<T>) {
               if (simde_mm512_cmpeq_epi8(rhs, simde_mm512_setzero_si512()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_epu8(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger8<T>) {
               if (simde_mm512_cmpeq_epi8(rhs, simde_mm512_setzero_si512()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_epi8(lhs, rhs);
            }
            else if constexpr (CT::UnsignedInteger16<T>) {
               if (simde_mm512_cmpeq_epi16(rhs, simde_mm512_setzero_si512()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_epu16(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger16<T>) {
               if (simde_mm512_cmpeq_epi16(rhs, simde_mm512_setzero_si512()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_epi16(lhs, rhs);
            }
            else if constexpr (CT::UnsignedInteger32<T>) {
               if (simde_mm512_cmpeq_epi32(rhs, simde_mm512_setzero_si512()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_epu32(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger32<T>) {
               if (simde_mm512_cmpeq_epi32(rhs, simde_mm512_setzero_si512()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_epi32(lhs, rhs);
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               if (simde_mm512_cmpeq_epi64(rhs, simde_mm512_setzero_si512()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_epu64(lhs, rhs);
            }
            else if constexpr (CT::SignedInteger64<T>) {
               if (simde_mm512_cmpeq_epi64(rhs, simde_mm512_setzero_si512()))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_epi64(lhs, rhs);
            }
            else if constexpr (CT::Float<T>) {
               if (simde_mm512_cmp_ps(rhs, simde_mm512_setzero_ps(), _CMP_EQ_OQ))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_ps(lhs, rhs);
            }
            else if constexpr (CT::Double<T>) {
               if (simde_mm512_cmp_pd(rhs, simde_mm512_setzero_pd(), _CMP_EQ_OQ))
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return simde_mm512_div_pd(lhs, rhs);
            }
            else
               LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else
      #endif
         LANGULUS_ERROR("Unsupported type");
      }

      /// Divide numbers at compile-time, if possible                         
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return array/scalar                                              
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto DivideConstexpr(const auto& lhsOrig, const auto& rhsOrig) {
         using DOUT = Decay<TypeOf<OUT>>;

         return Evaluate2<1, Unsupported, OUT>(
            lhsOrig, rhsOrig, nullptr,
            [](const DOUT& lhs, const DOUT& rhs) -> DOUT {
               if (rhs == DOUT {0})
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return lhs / rhs;
            }
         );
      }

      /// Divide numbers and return a register, if possible                   
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto Divide(const auto& lhsOrig, const auto& rhsOrig) {
         using DOUT = Decay<TypeOf<Desem<OUT>>>;
         using REGISTER = Register<decltype(lhsOrig), decltype(rhsOrig), OUT>;

         return Evaluate2<1, REGISTER, OUT>(
            lhsOrig, rhsOrig,
            [](const REGISTER& lhs, const REGISTER& rhs) {
               LANGULUS_SIMD_VERBOSE("Dividing (SIMD) as ", NameOf<REGISTER>());
               return DivideSIMD<DOUT>(lhs, rhs);
            },
            [](const DOUT& lhs, const DOUT& rhs) -> DOUT {
               LANGULUS_SIMD_VERBOSE("Dividing (Fallback) ", lhs, " / ", rhs, " (", NameOf<DOUT>(), ")");
               if (rhs == DOUT {0})
                  LANGULUS_THROW(DivisionByZero, "Division by zero");
               return lhs / rhs;
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

#include "IgnoreWarningsPop.inl"
