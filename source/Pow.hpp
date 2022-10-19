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

namespace Langulus::SIMD
{

   template<class T, Count S>
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
            if constexpr (CT::RealSP<T>)
               return simde_mm_pow_ps(lhs, rhs);
            else if constexpr (CT::RealDP<T>)
               return simde_mm_pow_pd(lhs, rhs);
            else if constexpr (CT::Integer32<T>) {
               auto result = simde_mm_set1_epi32(1);
               auto mask = simde_mm_cmpeq_epi32(rhs, simde_mm_setzero_si128());
               while (simde_mm_movemask_epi8(mask)) {
                  if constexpr (CT::Unsigned<T>) {
                     const auto onesmask = simde_mm_and_si128(rhs, simde_mm_set1_epi32(1));
                     if (simde_mm_movemask_epi8(onesmask)) { //TODO check performance without the branch
                        const auto tempr = simde_mm_mul_epu32(result, lhs);
                        result = simde_mm_castps_si128(simde_mm_blendv_ps(
                           simde_mm_castsi128_ps(result),
                           simde_mm_castsi128_ps(tempr),
                           simde_mm_castsi128_ps(simde_mm_and_si128(mask, onesmask))
                        ));
                     }
                     const auto temprhs = simde_mm_srli_epi32(rhs, 1);
                     rhs = simde_mm_castps_si128(simde_mm_blendv_ps(
                        simde_mm_castsi128_ps(rhs),
                        simde_mm_castsi128_ps(temprhs),
                        simde_mm_castsi128_ps(mask)
                     ));
                     const auto templhs = simde_mm_mul_epu32(lhs, lhs);
                     lhs = simde_mm_castps_si128(simde_mm_blendv_ps(
                        simde_mm_castsi128_ps(lhs),
                        simde_mm_castsi128_ps(templhs),
                        simde_mm_castsi128_ps(mask)
                     ));
                  }
                  else {
                     const auto tempr = simde_mm_mul_epi32(result, lhs);
                     result = simde_mm_castps_si128(simde_mm_blendv_ps(
                        simde_mm_castsi128_ps(result),
                        simde_mm_castsi128_ps(tempr),
                        simde_mm_castsi128_ps(mask)
                     ));
                     const auto temprhs = simde_mm_sub_epi32(rhs, simde_mm_set1_epi32(1));
                     rhs = simde_mm_castps_si128(simde_mm_blendv_ps(
                        simde_mm_castsi128_ps(rhs),
                        simde_mm_castsi128_ps(temprhs),
                        simde_mm_castsi128_ps(mask)
                     ));
                  }

                  mask = simde_mm_cmpeq_epi32(rhs, simde_mm_setzero_si128());
               }
               return result;
            }
            else if constexpr (CT::IntegerX<T>)
               return CT::Inner::NotSupported {};
            else LANGULUS_ERROR("Unsupported type for SIMD::PowerInner of 16-byte package");
         }
         else
      #endif
         
      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::RealSP<T>)
               return simde_mm256_pow_ps(lhs, rhs);
            else if constexpr (CT::RealDP<T>)
               return simde_mm256_pow_pd(lhs, rhs);
            else if constexpr (CT::Integer32<T>) {
               auto result = simde_mm256_set1_epi32(1);
               auto mask = simde_mm256_cmpeq_epi32(rhs, simde_mm256_setzero_si256());
               while (simde_mm256_movemask_epi8(mask)) {
                  if constexpr (CT::Unsigned<T>) {
                     const auto onesmask = simde_mm256_and_si128(rhs, simde_mm256_set1_epi32(1));
                     if (simde_mm256_movemask_epi8(onesmask)) { //TODO check performance without the branch
                        const auto tempr = simde_mm256_mul_epu32(result, lhs);
                        result = simde_mm256_castps_si256(simde_mm256_blendv_ps(
                           simde_mm256_castsi256_ps(result), 
                           simde_mm256_castsi256_ps(tempr),
                           simde_mm256_castsi256_ps(simde_mm256_and_si256(mask, onesmask))
                        ));
                     }
                     const auto temprhs = simde_mm256_srli_epi32(rhs, 1);
                     rhs = simde_mm256_castps_si256(simde_mm256_blendv_ps(
                        simde_mm256_castsi256_ps(rhs),
                        simde_mm256_castsi256_ps(temprhs),
                        simde_mm256_castsi256_ps(mask)
                     ));
                     const auto templhs = simde_mm256_mul_epu32(lhs, lhs);
                     lhs = simde_mm256_castps_si256(simde_mm256_blendv_ps(
                        simde_mm256_castsi256_ps(lhs),
                        simde_mm256_castsi256_ps(templhs),
                        simde_mm256_castsi256_ps(mask)
                     ));
                  }
                  else {
                     const auto tempr = simde_mm256_mul_epi32(result, lhs);
                     result = simde_mm256_castps_si256(simde_mm256_blendv_ps(
                        simde_mm256_castsi256_ps(result),
                        simde_mm256_castsi256_ps(tempr),
                        simde_mm256_castsi256_ps(mask)
                     ));
                     const auto temprhs = simde_mm256_sub_epi32(rhs, simde_mm256_set1_epi32(1));
                     rhs = simde_mm256_castps_si256(simde_mm256_blendv_ps(
                        simde_mm256_castsi256_ps(rhs),
                        simde_mm256_castsi256_ps(temprhs),
                        simde_mm256_castsi256_ps(mask)
                     ));
                  }

                  mask = simde_mm256_cmpeq_epi32(rhs, simde_mm256_setzero_si256());
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
            if constexpr (CT::RealSP<T>)
               return simde_mm512_pow_ps(lhs, rhs);
            else if constexpr (CT::RealDP<T>)
               return simde_mm512_pow_pd(lhs, rhs);
            else if constexpr (CT::Integer32<T>) {
               auto result = simde_mm512_set1_epi32(1);
               auto mask = simde_mm512_cmpeq_epi32(rhs, simde_mm512_setzero_si512());
               while (simde_mm512_movemask_epi8(mask)) {
                  if constexpr (CT::Unsigned<T>) {
                     const auto onesmask = simde_mm512_and_si128(rhs, simde_mm512_set1_epi32(1));
                     if (simde_mm512_movemask_epi8(onesmask)) { //TODO check performance without the branch
                        const auto tempr = simde_mm512_mul_epu32(result, lhs);
                        result = simde_mm512_blendv_ps(result, tempr, simde_mm512_and_si512(mask, onesmask));
                     }
                     const auto temprhs = simde_mm512_srli_epi32(rhs, 1);
                     rhs = simde_mm512_blendv_ps(rhs, temprhs, mask);
                     const auto templhs = simde_mm512_mul_epu32(lhs, lhs);
                     lhs = simde_mm512_blendv_ps(lhs, templhs, mask);
                  }
                  else {
                     const auto tempr = simde_mm512_mul_epi32(result, lhs);
                     result = simde_mm512_blendv_ps(result, tempr, mask);
                     const auto temprhs = simde_mm512_sub_epi32(rhs, simde_mm512_set1_epi32(1));
                     rhs = simde_mm512_blendv_ps(rhs, temprhs, mask);
                  }

                  mask = simde_mm512_cmpeq_epi32(rhs, simde_mm512_setzero_si512());
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
      return AttemptSIMD<0, REGISTER, LOSSLESS>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return PowerInner<LOSSLESS, S>(lhs, rhs);
         },
         [](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept -> LOSSLESS {
            return ::std::pow(lhs, rhs);
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