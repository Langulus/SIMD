///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Fill.hpp"
#include "Evaluate.hpp"
#include <cmath>


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<CT::Decayed, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported Power(const T&, const T&) noexcept {
         return {};
      }

      /// Raise by a power using SIMD                                         
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the raised values                                         
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto Power(UNUSED() REGISTER lhs, UNUSED() REGISTER rhs) noexcept {
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
                  return Unsupported {};
               else
                  LANGULUS_ERROR("Unsupported type for 16-byte package");
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
                  while (static_cast<uint32_t>(simde_mm256_movemask_epi8(mask)) != 0xFFFFFFFF) {
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
                  return Unsupported {};
               else
                  LANGULUS_ERROR("Unsupported type for 32-byte package");
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
                  TODO();
                  //https://stackoverflow.com/questions/42964882/test-if-a-big-integer-is-a-power-of-two
               }
               else if constexpr (CT::IntegerX<T>)
                  return Unsupported {};
               else
                  LANGULUS_ERROR("Unsupported type for 64-byte package");
            }
            else
         #endif
            LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Raise numbers to a power                                               
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<class LHS, class RHS, class OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   constexpr auto PowerConstexpr(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using DOUT = Decay<TypeOf<OUT>>;

      return Inner::Evaluate<1, Unsupported, OUT>(
         lhsOrig, rhsOrig, nullptr,
         [](DOUT lhs, DOUT rhs) noexcept -> DOUT {
            if (lhs == DOUT {1})
               return DOUT {1};

            if constexpr (CT::IntegerX<DOUT>) {
               if constexpr (CT::Unsigned<DOUT>) {
                  DOUT result {1};
                  while (rhs != DOUT {0}) {
                     if ((rhs & DOUT {1}) != DOUT {0})
                        result *= lhs;
                     rhs >>= DOUT {1};
                     lhs *= lhs;
                  }
                  return result;
               }
               else if (rhs > 0) {
                  DOUT result {1};
                  while (rhs != DOUT {0}) {
                     result *= lhs;
                     --rhs;
                  }
                  return result;
               }
               else return DOUT {0};
            }
            else if constexpr (CT::Real<DOUT>)
               return ::std::pow(lhs, rhs);
            else
               LANGULUS_ERROR("T must be a number");
         }
      );
   }
   
   /// Raise numbers to a power                                               
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<class LHS, class RHS, class OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   auto PowerDynamic(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using DOUT = Decay<TypeOf<OUT>>;
      using REGISTER = Inner::Register<LHS, RHS, OUT>;

      return Inner::Evaluate<1, REGISTER, OUT>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return Inner::Power<DOUT>(lhs, rhs);
         },
         [](DOUT lhs, DOUT rhs) noexcept -> DOUT {
            if (lhs == DOUT {1})
               return DOUT {1};

            if constexpr (CT::IntegerX<DOUT>) {
               if constexpr (CT::Unsigned<DOUT>) {
                  DOUT result {1};
                  while (rhs != DOUT {0}) {
                     if ((rhs & DOUT {1}) != DOUT {0})
                        result *= lhs;
                     rhs >>= DOUT {1};
                     lhs *= lhs;
                  }
                  return result;
               }
               else if (rhs > 0) {
                  DOUT result {1};
                  while (rhs != DOUT {0}) {
                     result *= lhs;
                     --rhs;
                  }
                  return result;
               }
               else return DOUT {0};
            }
            else if constexpr (CT::Real<DOUT>)
               return ::std::pow(lhs, rhs);
            else
               LANGULUS_ERROR("T must be a number");
         }
      );
   }

   /// Raise numbers to a power, and force output to desired place            
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<class LHS, class RHS, class OUT> LANGULUS(INLINED)
   constexpr void Power(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
      IF_CONSTEXPR() {
         StoreConstexpr(PowerConstexpr<LHS, RHS, OUT>(lhs, rhs), out);
      }
      else Store(PowerDynamic<LHS, RHS, OUT>(lhs, rhs), out);
   }

   /// Raise numbers to a power                                               
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<class LHS, class RHS, class OUT = std::array<Lossless<Decay<TypeOf<LHS>>, Decay<TypeOf<RHS>>>, OverlapCounts<LHS, RHS>()>>
   LANGULUS(INLINED)
   constexpr OUT Power(const LHS& lhs, const RHS& rhs) noexcept {
      OUT out;
      Power(lhs, rhs, out);
      return out;
   }

} // namespace Langulus::SIMD