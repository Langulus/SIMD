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
      constexpr Unsupported EqualsSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }
      
      /// Compare two registers                                               
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      R EqualsSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;

         if constexpr (CT::SIMD128<R>) {
            if      constexpr (CT::Integer8<T>)    return simde_mm_cmpeq_epi8    (lhs, rhs);
            else if constexpr (CT::Integer16<T>) {
               Logger::Info("simde_mm_cmpeq_epi16!!");
               return simde_mm_cmpeq_epi16(lhs, rhs);
            }
            else if constexpr (CT::Integer32<T>)   return simde_mm_cmpeq_epi32   (lhs, rhs);
            else if constexpr (CT::Integer64<T>)   return simde_mm_cmpeq_epi64   (lhs, rhs);
            else if constexpr (CT::Float<T>)       return simde_mm_cmpeq_ps      (lhs, rhs);
            else if constexpr (CT::Double<T>)      return simde_mm_cmpeq_pd      (lhs, rhs);
            else static_assert(false, "Unsupported type");
         }
         else if constexpr (CT::SIMD256<R>) {
            if      constexpr (CT::Integer8<T>)    return simde_mm256_cmpeq_epi8 (lhs, rhs);
            else if constexpr (CT::Integer16<T>)   return simde_mm256_cmpeq_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)   return simde_mm256_cmpeq_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)   return simde_mm256_cmpeq_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)       return simde_mm256_cmp_ps     (lhs, rhs, SIMDE_CMP_EQ_OQ);
            else if constexpr (CT::Double<T>)      return simde_mm256_cmp_pd     (lhs, rhs, SIMDE_CMP_EQ_OQ);
            else static_assert(false, "Unsupported type");
         }
         else if constexpr (CT::SIMD512<R>) {
            if      constexpr (CT::Integer8<T>)    return simde_mm512_cmpeq_epi8 (lhs, rhs);
            else if constexpr (CT::Integer16<T>)   return simde_mm512_cmpeq_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)   return simde_mm512_cmpeq_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)   return simde_mm512_cmpeq_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)       return simde_mm512_cmp_ps_mask(lhs, rhs, SIMDE_CMP_EQ_OQ);
            else if constexpr (CT::Double<T>)      return simde_mm512_cmp_pd_mask(lhs, rhs, SIMDE_CMP_EQ_OQ);
            else static_assert(false, "Unsupported type");
         }
         else static_assert(false, "Unsupported register");
      }
      
      /// Compare values as constexpr, if possible                            
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return bool/bitmask                                              
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto EqualsConstexpr(const auto& lhs, const auto& rhs) noexcept {
         // Will always return a std::array<bool>                       
         constexpr auto S = OverlapCounts<decltype(lhs), decltype(rhs)>();

         return AttemptBinary<0, ::std::array<bool, S>>(
            lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> bool {
               return l == r;
            }
         );
      }
   
      /// Compare values as a register, if possible                           
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return bool/bitmask/register                                     
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Equals(const auto& lhs, const auto& rhs) noexcept {
         // Will return either a std::array<bool>, or a masked register,
         // depending whether SIMD operation is supported or not        
         constexpr auto S = OverlapCounts<decltype(lhs), decltype(rhs)>();
         using OUT = Conditional<CT::SIMD<FORCE_OUT>, FORCE_OUT, ::std::array<bool, S>>;

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

   } // namespace Langulus::SIMD::Inner


   /// Compare numbers for equality, force output to desired place            
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in 'out'. Use Inner::Equals if you     
   ///      don't want this.                                                  
   template<class LHS, class RHS, CT::NoIntent OUT> LANGULUS(INLINED)
   constexpr void Equals(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
      IF_CONSTEXPR() {
         Store(Inner::EqualsConstexpr<OUT>(DeintCast(lhs), DeintCast(rhs)), out);
      }
      else {
         Store(Inner::Equals<OUT>(DeintCast(lhs), DeintCast(rhs)), out);
      }
   }

   /// Compare numbers for equality                                           
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in an instance of 'OUT'. Use           
   ///      Inner::Equals if you don't want this.                             
   template<class LHS, class RHS, CT::NoIntent OUT = Bitmask<OverlapCounts<LHS, RHS>()>>
   LANGULUS(INLINED)
   constexpr OUT Equals(const LHS& lhs, const RHS& rhs) noexcept {
      OUT out;
      Equals(DeintCast(lhs), DeintCast(rhs), out);
      return out;
   }

} // namespace Langulus::SIMD