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
      constexpr Unsupported Lesser(const T&, const T&) noexcept {
         return {};
      }
      
      /// Compare two arrays for lessness using SIMD                          
      ///   @tparam T - the type of the array element                         
      ///   @tparam S - the size of the array                                 
      ///   @tparam REGISTER - type of register we're operating with          
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return a bitmask with the results, or Inner::NotSupported        
      /// https://giannitedesco.github.io/2019/03/08/simd-cmp-bitmasks.html   
      template<CT::Decayed T, Count S, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto Lesser(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
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
               LANGULUS_ERROR("Unsupported type for 16-byte package");
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
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
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
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else
      #endif
         LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Compare numbers for lesser                                             
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (bitmask by default)          
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = Bitmask<OverlapCounts<LHS, RHS>()>>
   NOD() LANGULUS(INLINED)
   constexpr auto LesserConstexpr(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      // Output will likely contain a bool vector, or a bitmask         
      // so make sure we operate on Lossless<LHS, RHS>                  
      using DOUT = Decay<TypeOf<Lossless<LHS, RHS>>>;

      return Inner::Evaluate2<0, Unsupported, OUT>(
         lhsOrig, rhsOrig, nullptr,
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> bool {
            return lhs < rhs;
         }
      );
   }

   /// Compare numbers for lesser                                             
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (bitmask by default)          
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = Bitmask<OverlapCounts<LHS, RHS>()>>
   NOD() LANGULUS(INLINED)
   auto LesserDynamic(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      // Output will likely contain a bool vector, or a bitmask         
      // so make sure we operate on Lossless<LHS, RHS>                  
      using LOSSLESS = Lossless<LHS, RHS>;
      using DOUT = Decay<TypeOf<LOSSLESS>>;
      using REGISTER = Inner::Register<LHS, RHS, LOSSLESS>;
      constexpr auto S = OverlapCounts<LHS, RHS>();

      return Inner::Evaluate2<0, REGISTER, OUT>(
         lhsOrig, rhsOrig,
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return Inner::Lesser<DOUT, S>(lhs, rhs);
         },
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> bool {
            return lhs < rhs;
         }
      );
   }

   /// Compare numbers for lesser, and force output to desired place          
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT> LANGULUS(INLINED)
   constexpr void Lesser(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
      IF_CONSTEXPR() {
         StoreConstexpr(LesserConstexpr(lhs, rhs), out);
      }
      else Store(LesserDynamic(lhs, rhs), out);
   }

   /// Compare numbers for lesser                                             
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (defaults to bitmask)         
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = Bitmask<OverlapCounts<LHS, RHS>()>>
   LANGULUS(INLINED)
   constexpr OUT Lesser(const LHS& lhs, const RHS& rhs) noexcept {
      OUT out;
      Lesser(lhs, rhs, out);
      return out;
   }

} // namespace Langulus::SIMD