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
      constexpr Unsupported PowerSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Raise by power using registers                                      
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      auto PowerSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;

         if constexpr (CT::SIMD128<R>) {
            if constexpr (CT::Float<T>)            return R {simde_mm_pow_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)      return R {simde_mm_pow_pd(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger32<T>) {
               const auto one = simde_mm_set1_epi32(1);
               const auto zero = simde_mm_setzero_si128();

               R result = one;
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
            else if constexpr (CT::IntegerX<T>)    return Unsupported {};
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if constexpr (CT::Float<T>)            return R {simde_mm256_pow_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)      return R {simde_mm256_pow_pd(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger32<T>) {
               const auto one = simde_mm256_set1_epi32(1);
               const auto zero = simde_mm256_setzero_si256();

               R result = one;
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
            else if constexpr (CT::IntegerX<T>)    return Unsupported {};
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if constexpr (CT::Float<T>)                  return R {simde_mm512_pow_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)            return R {simde_mm512_pow_pd(lhs, rhs)};
            else if constexpr (CT::UnsignedInteger32<T>) {
               TODO();
               //https://stackoverflow.com/questions/42964882/test-if-a-big-integer-is-a-power-of-two
            }
            else if constexpr (CT::IntegerX<T>)          return Unsupported {};
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }
      
      /// Raise values to a power as constexpr, if possible                   
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the scalar/vector                                         
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto PowerConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(E l, E r) noexcept -> E {
               if (l == E {1})
                  return E {1};

               if constexpr (CT::IntegerX<E>) {
                  if constexpr (CT::Unsigned<E>) {
                     E result {1};
                     while (r != E {0}) {
                        if ((r & E {1}) != E {0})
                           result *= l;
                        r >>= E {1};
                        l *= l;
                     }
                     return result;
                  }
                  else if (r > 0) {
                     E result {1};
                     while (r != E {0}) {
                        result *= l;
                        --r;
                     }
                     return result;
                  }
                  else return E {0};
               }
               else if constexpr (CT::Real<E>)
                  return ::std::pow(l, r);
               else
                  LANGULUS_ERROR("T must be a number");
            }
         );
      }
   
      /// Raise values to a power as a register, if possible                  
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the scalar/vector/register                                
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Power(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Exponentiating (SIMD) as ", NameOf<R>());
               return PowerSIMD(l, r);
            },
            []<class E>(E l, E r) noexcept -> E {
               LANGULUS_SIMD_VERBOSE("Exponentiating (Fallback) ", l, " ^ ", r, " (", NameOf<E>(), ")");
               if (l == E {1})
                  return E {1};

               if constexpr (CT::IntegerX<E>) {
                  if constexpr (CT::Unsigned<E>) {
                     E result {1};
                     while (r != E {0}) {
                        if ((r & E {1}) != E {0})
                           result *= l;
                        r >>= E {1};
                        l *= l;
                     }
                     return result;
                  }
                  else if (r > 0) {
                     E result {1};
                     while (r != E {0}) {
                        result *= l;
                        --r;
                     }
                     return result;
                  }
                  else return E {0};
               }
               else if constexpr (CT::Real<E>)
                  return ::std::pow(l, r);
               else
                  LANGULUS_ERROR("T must be a number");
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(Power)

} // namespace Langulus::SIMD