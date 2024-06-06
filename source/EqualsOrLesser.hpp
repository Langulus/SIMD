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


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<CT::Decayed, Count, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported EqualsOrLesserSIMD(const T&, const T&) noexcept {
         return {};
      }
      
      /// Compare two arrays for equality-or-lesser using SIMD                
      ///   @tparam T - the type of the array element                         
      ///   @tparam S - the size of the array                                 
      ///   @tparam REGISTER - type of register we're operating with          
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return a bitmask with the results, or Inner::NotSupported        
      /// https://giannitedesco.github.io/2019/03/08/simd-cmp-bitmasks.html   
      template<CT::Decayed T, Count S, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto EqualsOrLesserSIMD(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm_cmple_epi8_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm_movemask_epi8(simde_mm_cmple_epi8(lhs, rhs))
                  };
               #endif
            }
            else if constexpr (CT::UnsignedInteger8<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm_cmple_epu8_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm_movemask_epi8(simde_mm_cmple_epi8(lhs, rhs))
                  };
               #endif
            }
            else if constexpr (CT::SignedInteger16<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm_cmple_epi16_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm_movemask_epi8(
                        simde_mm_packs_epi16(
                           simde_mm_cmple_epi16(lhs, rhs), 
                           simde_mm_setzero_si128()
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::UnsignedInteger16<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm_cmple_epu16_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm_movemask_epi8(
                        simde_mm_packs_epi16(
                           simde_mm_cmple_epi16(lhs, rhs), 
                           simde_mm_setzero_si128()
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::SignedInteger32<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm_cmple_epi32_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm_movemask_ps(
                        simde_mm_castsi128_ps(
                           simde_mm_cmple_epi32(lhs, rhs)
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::UnsignedInteger32<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm_cmple_epu32_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm_movemask_ps(
                        simde_mm_castsi128_ps(
                           simde_mm_cmple_epi32(lhs, rhs)
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::SignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm_cmple_epi64_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm_movemask_pd(
                        simde_mm_castsi128_pd(
                           simde_mm_cmple_epi64(lhs, rhs)
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm_cmple_epu64_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm_movemask_pd(
                        simde_mm_castsi128_pd(
                           simde_mm_cmple_epi64(lhs, rhs)
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::Float<T>)
               return Bitmask<S> {
                  simde_mm_movemask_ps(simde_mm_cmple_ps(lhs, rhs))
               };
            else if constexpr (CT::Double<T>)
               return Bitmask<S> {
                  simde_mm_movemask_pd(simde_mm_cmple_pd(lhs, rhs))
               };
            else
               LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm256_cmple_epi8_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm256_movemask_epi8(simde_mm256_cmple_epi8(lhs, rhs))
                  };
               #endif
            }
            else if constexpr (CT::UnsignedInteger8<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm256_cmple_epu8_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm256_movemask_epi8(simde_mm256_cmple_epi8(lhs, rhs))
                  };
               #endif
            }
            else if constexpr (CT::SignedInteger16<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm256_cmple_epi16_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm256_movemask_epi8(
                        lgls_pack_epi16(
                           simde_mm256_cmple_epi16(lhs, rhs), 
                           simde_mm256_setzero_si256()
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::UnsignedInteger16<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm256_cmple_epu16_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm256_movemask_epi8(
                        lgls_pack_epi16(
                           simde_mm256_cmple_epi16(lhs, rhs), 
                           simde_mm256_setzero_si256()
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::SignedInteger32<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm256_cmple_epi32_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm256_movemask_ps(
                        simde_mm256_castsi256_ps(
                           simde_mm256_cmple_epi32(lhs, rhs)
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::UnsignedInteger32<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm256_cmple_epu32_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm256_movemask_ps(
                        simde_mm256_castsi256_ps(
                           simde_mm256_cmple_epi32(lhs, rhs)
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::SignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm256_cmple_epi64_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm256_movemask_pd(
                        simde_mm256_castsi256_pd(
                           simde_mm256_cmple_epi64(lhs, rhs)
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return Bitmask<S> {
                     simde_mm256_cmple_epu64_mask(lhs, rhs)
                  };
               #else
                  return Bitmask<S> {
                     simde_mm256_movemask_pd(
                        simde_mm256_castsi256_pd(
                           simde_mm256_cmple_epi64(lhs, rhs)
                        )
                     )
                  };
               #endif
            }
            else if constexpr (CT::Float<T>) {
               return Bitmask<S> {
                  simde_mm256_movemask_ps(simde_mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ))
               };
            }
            else if constexpr (CT::Double<T>) {
               return Bitmask<S> {
                  simde_mm256_movemask_pd(simde_mm256_cmp_pd(lhs, rhs, _CMP_LE_OQ))
               };
            }
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>) {
               return Bitmask<S> {
                  simde_mm512_cmple_epi8_mask(lhs, rhs)
               };
            }
            else if constexpr (CT::UnsignedInteger8<T>) {
               return Bitmask<S> {
                  simde_mm512_cmple_epu8_mask(lhs, rhs)
               };
            }
            else if constexpr (CT::SignedInteger16<T>) {
               return Bitmask<S> {
                  simde_mm512_cmple_epi16_mask(lhs, rhs)
               };
            }
            else if constexpr (CT::UnsignedInteger16<T>) {
               return Bitmask<S> {
                  simde_mm512_cmple_epu16_mask(lhs, rhs)
               };
            }
            else if constexpr (CT::SignedInteger32<T>) {
               return Bitmask<S> {
                  simde_mm512_cmple_epi32_mask(lhs, rhs)
               };
            }
            else if constexpr (CT::UnsignedInteger32<T>) {
               return Bitmask<S> {
                  simde_mm512_cmple_epu32_mask(lhs, rhs)
               };
            }
            else if constexpr (CT::SignedInteger64<T>) {
               return Bitmask<S> {
                  simde_mm512_cmple_epi64_mask(lhs, rhs)
               };
            }
            else if constexpr (CT::UnsignedInteger64<T>) {
               return Bitmask<S> {
                  simde_mm512_cmple_epu64_mask(lhs, rhs)
               };
            }
            else if constexpr (CT::Float<T>) {
               return Bitmask<S> {
                  simde_mm512_cmp_ps_mask(lhs, rhs, _CMP_LE_OQ)
               };
            }
            else if constexpr (CT::Double<T>) {
               return Bitmask<S> {
                  simde_mm512_cmp_pd_mask(lhs, rhs, _CMP_LE_OQ)
               };
            }
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else
      #endif
         LANGULUS_ERROR("Unsupported type");
      }
      
      /// Compare numbers for equal-or-less at compile-time, if possible      
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return array/scalar                                              
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto EqualsOrLesserConstexpr(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         // Output will likely contain a bool vector, or a bitmask      
         // so make sure we operate on Lossless<LHS, RHS>               
         using DOUT = Decay<TypeOf<Lossless<decltype(lhsOrig), decltype(rhsOrig)>>>;

         return Inner::Evaluate2<0, Unsupported, OUT>(
            lhsOrig, rhsOrig, nullptr,
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> bool {
               return lhs <= rhs;
            }
         );
      }
   
      /// Compare numbers for equal-or-less and return a bitmask              
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///      or array/scalar if no viable SIMD routine exists               
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto EqualsOrLesser(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         using LOSSLESS = Lossless<decltype(lhsOrig), decltype(rhsOrig)>;
         using DOUT = Decay<TypeOf<LOSSLESS>>;
         using REGISTER = Inner::Register<decltype(lhsOrig), decltype(rhsOrig), LOSSLESS>;
         constexpr auto S = OverlapCounts<decltype(lhsOrig), decltype(rhsOrig)>();

         return Evaluate2<0, REGISTER, OUT>(
            lhsOrig, rhsOrig,
            [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
               LANGULUS_SIMD_VERBOSE("Comparing for equal-or-lesser (SIMD) as ", NameOf<REGISTER>());
               return EqualsOrLesserSIMD<DOUT, S>(lhs, rhs);
            },
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> bool {
               LANGULUS_SIMD_VERBOSE("Comparing for equal-or-lesser (Fallback) ", lhs, " <= ", rhs, " (", NameOf<DOUT>(), ")");
               return lhs <= rhs;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   /// Compare numbers for equal-or-less, force output to desired place       
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in 'out'. Use Inner::EqualsOrGreater   
   ///      if you don't want this.                                           
   template<class LHS, class RHS, CT::NotSemantic OUT> LANGULUS(INLINED)
   constexpr void EqualsOrLesser(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
      if consteval {
         Store(Inner::EqualsOrLesserConstexpr<OUT>(DesemCast(lhs), DesemCast(rhs)), out);
      }
      else {
         Store(Inner::EqualsOrLesser<OUT>(DesemCast(lhs), DesemCast(rhs)), out);
      }
   }

   /// Compare numbers for equal-or-less                                      
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in an instance of 'OUT'. Use           
   ///      Inner::EqualsOrGreater if you don't want this.                    
   template<class LHS, class RHS, CT::NotSemantic OUT = Bitmask<OverlapCounts<LHS, RHS>()>>
   LANGULUS(INLINED)
   constexpr OUT EqualsOrLesser(const LHS& lhs, const RHS& rhs) noexcept {
      OUT out;
      EqualsOrLesser(DesemCast(lhs), DesemCast(rhs), out);
      return out;
   }

} // namespace Langulus::SIMD