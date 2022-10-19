///                                                                           
/// Langulus::SIMD                                                            
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>                    
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Fill.hpp"
#include "Convert.hpp"
#include "IgnoreWarningsPush.inl"

namespace Langulus::SIMD
{

   template<class, Count>
   LANGULUS(ALWAYSINLINE) constexpr auto DivideInner(CT::NotSupported auto, CT::NotSupported auto) noexcept {
      return CT::Inner::NotSupported{};
   }

   /// Divide two arrays using SIMD                                           
   ///   @tparam T - the type of the array element                            
   ///   @tparam S - the size of the array                                    
   ///   @tparam REGISTER - type of register we're operating with             
   ///   @param lhs - the left-hand-side array                                
   ///   @param rhs - the right-hand-side array                               
   ///   @return the divided elements as a register                           
   template<class T, Count S, CT::TSIMD REGISTER>
   LANGULUS(ALWAYSINLINE) auto DivideInner(const REGISTER& lhs, const REGISTER& rhs) {
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
         else if constexpr (CT::RealSP<T>) {
            if (simde_mm_movemask_ps(simde_mm_cmpeq_ps(rhs, simde_mm_setzero_ps())))
               LANGULUS_THROW(DivisionByZero, "Division by zero");
            return simde_mm_div_ps(lhs, rhs);
         }
         else if constexpr (CT::RealDP<T>) {
            if (simde_mm_movemask_pd(simde_mm_cmpeq_pd(rhs, simde_mm_setzero_pd())))
               LANGULUS_THROW(DivisionByZero, "Division by zero");
            return simde_mm_div_pd(lhs, rhs);
         }
         else LANGULUS_ERROR("Unsupported type for SIMD::DivideInner of 16-byte package");
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
         else if constexpr (CT::RealSP<T>) {
            if (simde_mm256_movemask_ps(simde_mm256_cmp_ps(rhs, simde_mm256_setzero_ps(), _CMP_EQ_OQ)))
               LANGULUS_THROW(DivisionByZero, "Division by zero");
            return simde_mm256_div_ps(lhs, rhs);
         }
         else if constexpr (CT::RealDP<T>) {
            if (simde_mm256_movemask_pd(simde_mm256_cmp_pd(rhs, simde_mm256_setzero_pd(), _CMP_EQ_OQ)))
               LANGULUS_THROW(DivisionByZero, "Division by zero");
            return simde_mm256_div_pd(lhs, rhs);
         }
         else LANGULUS_ERROR("Unsupported type for SIMD::DivideInner of 32-byte package");
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
         else if constexpr (CT::RealSP<T>) {
            if (simde_mm512_cmp_ps(rhs, simde_mm512_setzero_ps(), _CMP_EQ_OQ))
               LANGULUS_THROW(DivisionByZero, "Division by zero");
            return simde_mm512_div_ps(lhs, rhs);
         }
         else if constexpr (CT::RealDP<T>) {
            if (simde_mm512_cmp_pd(rhs, simde_mm512_setzero_pd(), _CMP_EQ_OQ))
               LANGULUS_THROW(DivisionByZero, "Division by zero");
            return simde_mm512_div_pd(lhs, rhs);
         }
         else
            LANGULUS_ERROR("Unsupported type for SIMD::DivideInner of 64-byte package");
      }
      else
   #endif

      LANGULUS_ERROR("Unsupported type for SIMD::DivideInner");
   }

   ///                                                                        
   template<class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) auto Divide(const LHS& lhsOrig, const RHS& rhsOrig) {
      using REGISTER = CT::Register<LHS, RHS>;
      using LOSSLESS = Lossless<LHS, RHS>;
      constexpr auto S = OverlapCount<LHS, RHS>();
      return AttemptSIMD<1, REGISTER, LOSSLESS>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) {
            return DivideInner<LOSSLESS, S>(lhs, rhs);
         },
         [](const LOSSLESS& lhs, const LOSSLESS& rhs) -> LOSSLESS {
            if (rhs == LOSSLESS {0})
               LANGULUS_THROW(DivisionByZero, "Division by zero");
            if constexpr (CT::Same<LOSSLESS, ::std::byte>) {
               // ::std::byte doesn't have * operator                   
               return static_cast<LOSSLESS>(
                  reinterpret_cast<const unsigned char&>(lhs) /
                  reinterpret_cast<const unsigned char&>(rhs)
               );
            }
            else return lhs / rhs;
         }
      );
   }

   ///                                                                        
   template<class LHS, class RHS, class OUT>
   LANGULUS(ALWAYSINLINE) void Divide(const LHS& lhs, const RHS& rhs, OUT& output) {
      GeneralStore(Divide<LHS, RHS>(lhs, rhs), output);
   }

   ///                                                                        
   template<CT::Vector WRAPPER, class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) WRAPPER DivideWrap(const LHS& lhs, const RHS& rhs) {
      WRAPPER result;
      Divide<LHS, RHS>(lhs, rhs, result.mArray);
      return result;
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
