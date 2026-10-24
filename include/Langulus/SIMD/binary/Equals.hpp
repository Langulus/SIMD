///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "../Attempt.hpp"


namespace Langulus::SIMD::Inner
{
   /// Used to detect missing SIMD routine                                    
   LANGULUS(INLINED)
   constexpr No EqualsSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept { return {}; }
   
   /// Compare two registers using SIMD                                       
   template<CT::SIMD R> LANGULUS(INLINED)
   R EqualsSIMD(R lhs, R rhs) noexcept {
      using T = TypeOf<R>;
      (void)lhs; (void)rhs;

   #if LANGULUS_SIMD(128BIT)
      if constexpr (CT::SIMD128<R>) {
         if      constexpr (CT::Integer8<T>)    return simde_mm_cmpeq_epi8    (lhs, rhs);
         else if constexpr (CT::Integer16<T>)   return simde_mm_cmpeq_epi16   (lhs, rhs);
         else if constexpr (CT::Integer32<T>)   return simde_mm_cmpeq_epi32   (lhs, rhs);
         else if constexpr (CT::Integer64<T>)   return simde_mm_cmpeq_epi64   (lhs, rhs);
         else if constexpr (CT::Real32<T>)      return simde_mm_cmpeq_ps      (lhs, rhs);
         else if constexpr (CT::Real64<T>)      return simde_mm_cmpeq_pd      (lhs, rhs);
         else static_assert(false, "Unsupported type");
      }
      else
   #endif
   #if LANGULUS_SIMD(256BIT)
      if constexpr (CT::SIMD256<R>) {
         if      constexpr (CT::Integer8<T>)    return simde_mm256_cmpeq_epi8 (lhs, rhs);
         else if constexpr (CT::Integer16<T>)   return simde_mm256_cmpeq_epi16(lhs, rhs);
         else if constexpr (CT::Integer32<T>)   return simde_mm256_cmpeq_epi32(lhs, rhs);
         else if constexpr (CT::Integer64<T>)   return simde_mm256_cmpeq_epi64(lhs, rhs);
         else if constexpr (CT::Real32<T>)      return simde_mm256_cmp_ps     (lhs, rhs, SIMDE_CMP_EQ_OQ);
         else if constexpr (CT::Real64<T>)      return simde_mm256_cmp_pd     (lhs, rhs, SIMDE_CMP_EQ_OQ);
         else static_assert(false, "Unsupported type");
      }
      else
   #endif
   #if LANGULUS_SIMD(512BIT)
      if constexpr (CT::SIMD512<R>) {
         if      constexpr (CT::Integer8<T>)    return simde_mm512_cmpeq_epi8 (lhs, rhs);
         else if constexpr (CT::Integer16<T>)   return simde_mm512_cmpeq_epi16(lhs, rhs);
         else if constexpr (CT::Integer32<T>)   return simde_mm512_cmpeq_epi32(lhs, rhs);
         else if constexpr (CT::Integer64<T>)   return simde_mm512_cmpeq_epi64(lhs, rhs);
         else if constexpr (CT::Real32<T>)      return simde_mm512_cmp_ps_mask(lhs, rhs, SIMDE_CMP_EQ_OQ);
         else if constexpr (CT::Real64<T>)      return simde_mm512_cmp_pd_mask(lhs, rhs, SIMDE_CMP_EQ_OQ);
         else static_assert(false, "Unsupported type");
      }
      else
   #endif
      static_assert(false, "Unsupported register");
   }
   
   /// Compare values as constexpr, if possible                               
   ///   @tparam FORCE_OUT the desired element type (lossless if void)        
   ///   @return bool/bitmask                                                 
   template<class FORCE_OUT = void> LANGULUS(INLINED)
   constexpr auto EqualsConstexpr(auto const& lhs, auto const& rhs) noexcept {
      // Will always return a std::array<bool>                          
      constexpr size_t S = LglsOverlapExtents(lhs, rhs);

      return AttemptBinary<0, ::std::array<bool, S>>(
         lhs, rhs, nullptr,
         []<class E>(const E& l, const E& r) noexcept -> bool {
            return l == r;
         }
      );
   }

   /// Compare values as a register, if possible                              
   ///   @tparam FORCE_OUT the desired element type (lossless if void)        
   ///   @param value scalar/vector/register to operate on                    
   ///   @return bool/bitmask/register                                        
   template<class FORCE_OUT = void> LANGULUS(INLINED)
   auto Equals(auto const& lhs, auto const& rhs) noexcept {
      // Will return either a std::array<bool>, or a masked register,   
      // depending whether SIMD operation is supported or not           
      constexpr size_t S = LglsOverlapExtents(lhs, rhs);
      using OUT = Tif<CT::SIMD<FORCE_OUT>, FORCE_OUT, ::std::array<bool, S>>;

      return AttemptBinary<0, OUT>(lhs, rhs,
         []<class R>(const R& l, const R& r) noexcept {
            LANGULUS_SIMD_VERBOSE("Comparing for equality (SIMD) as ", NameOf<R>());
            return EqualsSIMD(l, r);
         },
         []<class E>(const E& l, const E& r) noexcept -> bool {
            LANGULUS_SIMD_VERBOSE("Comparing for equality (Fallback) ", l, " == ", r, " (", NameOf<E>(), ")");
            return l == r;
         }
      );
   }
}

namespace Langulus::SIMD
{
   /// Compare numbers for equality, force output to desired place            
   ///   @param lhs left array, scalar, or register                           
   ///   @param rhs right array, scalar, or register                          
   ///   @param out [out] output                                              
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in 'out'. Use Inner::Equals if you     
   ///      don't want this.                                                  
   template<class LHS, class RHS, class OUT>
   requires CT::NoIntent<LHS, RHS, OUT> LANGULUS(INLINED)
   constexpr void Equals(LHS const& lhs, RHS const& rhs, OUT& out) noexcept {
      if consteval {
         Store(Inner::EqualsConstexpr<OUT>(lhs, rhs), out);
      }
      else {
         Store(Inner::Equals<OUT>(lhs, rhs), out);
      }
   }

   /// Compare numbers for equality                                           
   ///   @param lhs left array, scalar, or register                           
   ///   @param rhs right array, scalar, or register                          
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in an instance of 'OUT'. Use           
   ///      Inner::Equals if you don't want this.                             
   template<class LHS, class RHS, class OUT = BitmaskFromOverlap<LHS, RHS>>
   requires CT::NoIntent<LHS, RHS, OUT> LANGULUS(INLINED)
   constexpr OUT Equals(LHS const& lhs, RHS const& rhs) noexcept {
      OUT out;
      Equals(lhs, rhs, out);
      return out;
   }
}