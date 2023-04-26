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

   template<class, Count>
   LANGULUS(INLINED) constexpr auto LesserInner(CT::Unsupported auto, CT::Unsupported auto) noexcept {
      return Unsupported{};
   }
      
   /// Compare two arrays for lessness using SIMD                             
   ///   @tparam T - the type of the array element                            
   ///   @tparam S - the size of the array                                    
   ///   @tparam REGISTER - type of register we're operating with             
   ///   @param lhs - the left-hand-side array                                
   ///   @param rhs - the right-hand-side array                               
   ///   @return a bitmask with the results, or Inner::NotSupported           
   /// https://giannitedesco.github.io/2019/03/08/simd-cmp-bitmasks.html      
   template<class T, Count S, CT::TSIMD REGISTER>
   LANGULUS(INLINED) auto LesserInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
   #if LANGULUS_SIMD(128BIT)
      if constexpr (CT::SIMD128<REGISTER>) {
         if constexpr (CT::SignedInteger8<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm_cmplt_epi8_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm_movemask_epi8(simde_mm_cmplt_epi8(lhs, rhs))
               };
            #endif
         }
         else if constexpr (CT::UnsignedInteger8<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm_cmplt_epu8_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm_movemask_epi8(simde_mm_cmplt_epi8(lhs, rhs))
               };
            #endif
         }
         else if constexpr (CT::SignedInteger16<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm_cmplt_epi16_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm_movemask_epi8(
                     simde_mm_packs_epi16(
                        simde_mm_cmplt_epi16(lhs, rhs), 
                        simde_mm_setzero_si128()
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm_cmplt_epu16_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm_movemask_epi8(
                     simde_mm_packs_epi16(
                        simde_mm_cmplt_epi16(lhs, rhs), 
                        simde_mm_setzero_si128()
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::SignedInteger32<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm_cmplt_epi32_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm_movemask_ps(
                     simde_mm_castsi128_ps(
                        simde_mm_cmplt_epi32(lhs, rhs)
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::UnsignedInteger32<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm_cmplt_epu32_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm_movemask_ps(
                     simde_mm_castsi128_ps(
                        simde_mm_cmplt_epi32(lhs, rhs)
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::SignedInteger64<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm_cmplt_epi64_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm_movemask_pd(
                     simde_mm_castsi128_pd(
                        simde_mm_cmplt_epi64(lhs, rhs)
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm_cmplt_epu64_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm_movemask_pd(
                     simde_mm_castsi128_pd(
                        simde_mm_cmplt_epi64(lhs, rhs)
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::Float<T>)
            return Bitmask<S> {
               simde_mm_movemask_ps(simde_mm_cmplt_ps(lhs, rhs))
            };
         else if constexpr (CT::Double<T>)
            return Bitmask<S> {
               simde_mm_movemask_pd(simde_mm_cmplt_pd(lhs, rhs))
            };
         else
            LANGULUS_ERROR("Unsupported type for SIMD::LesserInner of 16-byte package");
      }
      else
   #endif

   #if LANGULUS_SIMD(256BIT)
      if constexpr (CT::SIMD256<REGISTER>) {
         if constexpr (CT::SignedInteger8<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm256_cmplt_epi8_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm256_movemask_epi8(simde_mm256_cmplt_epi8(lhs, rhs))
               };
            #endif
         }
         else if constexpr (CT::UnsignedInteger8<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm256_cmplt_epu8_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm256_movemask_epi8(simde_mm256_cmplt_epi8(lhs, rhs))
               };
            #endif
         }
         else if constexpr (CT::SignedInteger16<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm256_cmplt_epi16_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm256_movemask_epi8(
                     lgls_pack_epi16(
                        simde_mm256_cmplt_epi16(lhs, rhs), 
                        simde_mm256_setzero_si256()
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm256_cmplt_epu16_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm256_movemask_epi8(
                     lgls_pack_epi16(
                        simde_mm256_cmplt_epi16(lhs, rhs), 
                        simde_mm256_setzero_si256()
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::SignedInteger32<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm256_cmplt_epi32_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm256_movemask_ps(
                     simde_mm256_castsi256_ps(
                        simde_mm256_cmplt_epi32(lhs, rhs)
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::UnsignedInteger32<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm256_cmplt_epu32_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm256_movemask_ps(
                     simde_mm256_castsi256_ps(
                        simde_mm256_cmplt_epi32(lhs, rhs)
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::SignedInteger64<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm256_cmplt_epi64_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm256_movemask_pd(
                     simde_mm256_castsi256_pd(
                        simde_mm256_cmplt_epi64(lhs, rhs)
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<T>) {
            #if LANGULUS_SIMD(AVX512)
               return Bitmask<S> {
                  simde_mm256_cmplt_epu64_mask(lhs, rhs)
               };
            #else
               return Bitmask<S> {
                  simde_mm256_movemask_pd(
                     simde_mm256_castsi256_pd(
                        simde_mm256_cmplt_epi64(lhs, rhs)
                     )
                  )
               };
            #endif
         }
         else if constexpr (CT::Float<T>) {
            return Bitmask<S> {
               simde_mm256_movemask_ps(simde_mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ))
            };
         }
         else if constexpr (CT::Double<T>) {
            return Bitmask<S> {
               simde_mm256_movemask_pd(simde_mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ))
            };
         }
         else LANGULUS_ERROR("Unsupported type for SIMD::LesserInner of 32-byte package");
      }
      else
   #endif

   #if LANGULUS_SIMD(512BIT)
      if constexpr (CT::SIMD512<REGISTER>) {
         if constexpr (CT::SignedInteger8<T>) {
            return Bitmask<S> {
               simde_mm512_cmplt_epi8_mask(lhs, rhs)
            };
         }
         else if constexpr (CT::UnsignedInteger8<T>) {
            return Bitmask<S> {
               simde_mm512_cmplt_epu8_mask(lhs, rhs)
            };
         }
         else if constexpr (CT::SignedInteger16<T>) {
            return Bitmask<S> {
               simde_mm512_cmplt_epi16_mask(lhs, rhs)
            };
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            return Bitmask<S> {
               simde_mm512_cmplt_epu16_mask(lhs, rhs)
            };
         }
         else if constexpr (CT::SignedInteger32<T>) {
            return Bitmask<S> {
               simde_mm512_cmplt_epi32_mask(lhs, rhs)
            };
         }
         else if constexpr (CT::UnsignedInteger32<T>) {
            return Bitmask<S> {
               simde_mm512_cmplt_epu32_mask(lhs, rhs)
            };
         }
         else if constexpr (CT::SignedInteger64<T>) {
            return Bitmask<S> {
               simde_mm512_cmplt_epi64_mask(lhs, rhs)
            };
         }
         else if constexpr (CT::UnsignedInteger64<T>) {
            return Bitmask<S> {
               simde_mm512_cmplt_epu64_mask(lhs, rhs)
            };
         }
         else if constexpr (CT::Float<T>) {
            return Bitmask<S> {
               simde_mm512_cmp_ps_mask(lhs, rhs, _CMP_LT_OQ)
            };
         }
         else if constexpr (CT::Double<T>) {
            return Bitmask<S> {
               simde_mm512_cmp_pd_mask(lhs, rhs, _CMP_LT_OQ)
            };
         }
         else LANGULUS_ERROR("Unsupported type for SIMD::LesserInner of 64-byte package");
      }
      else
   #endif
      LANGULUS_ERROR("Unsupported type for SIMD::LesserInner");
   }

   ///                                                                        
   template<class LHS, class RHS>
   NOD() LANGULUS(INLINED) auto Lesser(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using LOSSLESS = Lossless<LHS, RHS>;
      using REGISTER = CT::Register<LHS, RHS, LOSSLESS>;
      constexpr auto S = OverlapCount<LHS, RHS>();

      return AttemptSIMD<0, REGISTER, LOSSLESS>(
         lhsOrig, rhsOrig,
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return LesserInner<LOSSLESS, S>(lhs, rhs);
         },
         [](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept -> bool {
            return lhs < rhs;
         }
      );
   }

   ///                                                                        
   template<class LHS, class RHS, class OUT>
   LANGULUS(INLINED) void Lesser(const LHS& lhs, const RHS& rhs, OUT& output) noexcept {
      GeneralStore(Lesser<LHS, RHS>(lhs, rhs), output);
   }

   ///                                                                        
   template<CT::Vector WRAPPER, class LHS, class RHS>
   NOD() LANGULUS(INLINED) WRAPPER LesserWrap(const LHS& lhs, const RHS& rhs) noexcept {
      WRAPPER result;
      Lesser<LHS, RHS>(lhs, rhs, result.mArray);
      return result;
   }

} // namespace Langulus::SIMD