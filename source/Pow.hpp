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
#include <cmath>

namespace Langulus::SIMD
{

   template<class, Count>
   LANGULUS(ALWAYSINLINE) constexpr auto PowerInner(CT::NotSupported auto, CT::NotSupported auto) noexcept {
      return CT::Inner::NotSupported{};
   }

   /// Raise by a power using SIMD                                            
   ///   @tparam T - the type of the array element                            
   ///   @tparam S - the size of the array                                    
   ///   @tparam REGISTER - the register type (deducible)                     
   ///   @param lhs - the left-hand-side array                                
   ///   @param rhs - the right-hand-side array                               
   ///   @return the raised values                                            
   template<class T, Count S, CT::TSIMD REGISTER>
   LANGULUS(ALWAYSINLINE) auto PowerInner(REGISTER lhs, REGISTER rhs) noexcept {
      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm_pow_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm_pow_pd(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>) {
               const auto one = simde_mm_set1_epi32(1);
               const auto zero = simde_mm_setzero_si128();

               auto result = one;
               auto mask = simde_mm_cmpeq_epi32(rhs, zero);
               while (simde_mm_movemask_epi8(mask) != 0xFFFF) {
                  const auto onesmask = simde_mm_cmpeq_epi32(simde_mm_and_si128(rhs, one), one);
                 // if (simde_mm_movemask_epi8(onesmask)) { //TODO check performance with the branch
                     const auto tempr = simde_mm_mullo_epi32(result, lhs);
                     const auto mixer = simde_mm_andnot_si128(mask, onesmask);
                     result = lgls_blendv_epi32(result, tempr, mixer);
                 // }
                  const auto temprhs = simde_mm_srli_epi32(rhs, 1);
                  rhs = lgls_blendv_epi32(temprhs, rhs, mask);
                  const auto templhs = simde_mm_mullo_epi32(lhs, lhs);
                  lhs = lgls_blendv_epi32(templhs, lhs, mask);

                  mask = simde_mm_cmpeq_epi32(rhs, zero);
               }

               return result;
            }
            else if constexpr (CT::IntegerX<T>)
               return CT::Inner::NotSupported {};
            else
               LANGULUS_ERROR("Unsupported type for SIMD::PowerInner of 16-byte package");
         }
         else
      #endif
         
      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm256_pow_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm256_pow_pd(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>) {
               const auto one = simde_mm256_set1_epi32(1);
               const auto zero = simde_mm256_setzero_si256();

               auto result = one;
               auto mask = simde_mm256_cmpeq_epi32(rhs, zero);
               while (simde_mm256_movemask_epi8(mask) != 0xFFFFFFFF) {
                  const auto onesmask = simde_mm256_cmpeq_epi32(simde_mm256_and_si256(rhs, one), one);
                 // if (simde_mm256_movemask_epi8(onesmask)) { //TODO check performance with the branch
                     const auto tempr = simde_mm256_mullo_epi32(result, lhs);
                     const auto mixer = simde_mm256_andnot_si256(mask, onesmask);
                     result = lgls_blendv_epi32(result, tempr, mixer);
                 // }
                  const auto temprhs = simde_mm256_srli_epi32(rhs, 1);
                  rhs = lgls_blendv_epi32(temprhs, rhs, mask);
                  const auto templhs = simde_mm256_mullo_epi32(lhs, lhs);
                  lhs = lgls_blendv_epi32(templhs, lhs, mask);

                  mask = simde_mm256_cmpeq_epi32(rhs, zero);
               }
               return result;
            }
            else if constexpr (CT::IntegerX<T>)
               return CT::Inner::NotSupported {};
            else
               LANGULUS_ERROR("Unsupported type for SIMD::PowerInner of 32-byte package");
         }
         else
      #endif
            
      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm512_pow_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm512_pow_pd(lhs, rhs);
            else if constexpr (CT::UnsignedInteger32<T>) {
               const auto one = simde_mm512_set1_epi32(1);
               const auto zero = simde_mm512_setzero_si512();

               auto result = one;
               auto mask = simde_mm512_cmpeq_epi32(rhs, zero);
               while (simde_mm512_movemask_epi8(mask) != 0xFFFFFFFF) {
                  const auto onesmask = simde_mm512_cmpeq_epi32(simde_mm512_and_si128(rhs, one), one);
                  //if (simde_mm512_movemask_epi8(onesmask)) { //TODO check performance with the branch
                     const auto tempr = simde_mm512_mullo_epi32(result, lhs);
                     const auto mixer = simde_mm512_andnot_si512(mask, onesmask);
                     result = lgls_blendv_epi32(result, tempr, mixer);
                  //}
                  const auto temprhs = simde_mm512_srli_epi32(rhs, 1);
                  rhs = lgls_blendv_epi32(temprhs, rhs, mask);
                  const auto templhs = simde_mm512_mullo_epi32(lhs, lhs);
                  lhs = lgls_blendv_epi32(templhs, lhs, mask);

                  mask = simde_mm512_cmpeq_epi32(rhs, zero);
               }
               return result;
            }
            else if constexpr (CT::IntegerX<T>)
               return CT::Inner::NotSupported {};
            else
               LANGULUS_ERROR("Unsupported type for SIMD::PowerInner of 64-byte package");
         }
         else
      #endif
         LANGULUS_ERROR("Unsupported type for SIMD::PowerInner");
   }

   ///                                                                        
   template<class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) auto Power(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using REGISTER = CT::Register<LHS, RHS>;
      using LOSSLESS = Lossless<LHS, RHS>;
      constexpr auto S = OverlapCount<LHS, RHS>();
      return AttemptSIMD<1, REGISTER, LOSSLESS>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return PowerInner<LOSSLESS, S>(lhs, rhs);
         },
         [](LOSSLESS lhs, LOSSLESS rhs) noexcept -> LOSSLESS {
            if (lhs == LOSSLESS {1})
               return LOSSLESS {1};

            if constexpr (CT::IntegerX<LOSSLESS>) {
               if constexpr (CT::Unsigned<LOSSLESS>) {
                  LOSSLESS result {1};
                  while (rhs != LOSSLESS {0}) {
                     if ((rhs & LOSSLESS {1}) != LOSSLESS {0})
                        result *= lhs;
                     rhs >>= LOSSLESS {1};
                     lhs *= lhs;
                  }
                  return result;
               }
               else if (rhs > 0) {
                  LOSSLESS result {1};
                  while (rhs != LOSSLESS {0}) {
                     result *= lhs;
                     --rhs;
                  }
                  return result;
               }
               else return LOSSLESS {0};
            }
            else if constexpr (CT::Real<LOSSLESS>)
               return ::std::pow(lhs, rhs);
            else
               LANGULUS_ERROR("T must be a number");
         }
      );
   }

   ///                                                                        
   template<class LHS, class RHS, class OUT>
   LANGULUS(ALWAYSINLINE) void Power(const LHS& lhs, const RHS& rhs, OUT& output) noexcept {
      GeneralStore(Power<LHS, RHS>(lhs, rhs), output);
   }

   ///                                                                        
   template<CT::Vector WRAPPER, class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) WRAPPER PowerWrap(const LHS& lhs, const RHS& rhs) noexcept {
      WRAPPER result;
      Power<LHS, RHS>(lhs, rhs, result.mArray);
      return result;
   }

} // namespace Langulus::SIMD