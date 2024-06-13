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
      constexpr Unsupported GreaterSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }
      
      /// Compare two registers                                               
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      R GreaterSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;
         
         if constexpr (CT::SIMD128<R>) {
            if constexpr (CT::Integer8<T>)         return simde_mm_cmpgt_epi8(lhs, rhs);
            else if constexpr (CT::Integer16<T>)   return simde_mm_cmpgt_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)   return simde_mm_cmpgt_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)   return simde_mm_cmpgt_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)       return simde_mm_cmpgt_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)      return simde_mm_cmpgt_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if constexpr (CT::Integer8<T>)         return simde_mm256_cmpgt_epi8(lhs, rhs);
            else if constexpr (CT::Integer16<T>)   return simde_mm256_cmpgt_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)   return simde_mm256_cmpgt_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)   return simde_mm256_cmpgt_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)       return simde_mm256_cmpgt_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)      return simde_mm256_cmpgt_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if constexpr (CT::Integer8<T>)         return simde_mm512_cmpgt_epi8(lhs, rhs);
            else if constexpr (CT::Integer16<T>)   return simde_mm512_cmpgt_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)   return simde_mm512_cmpgt_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)   return simde_mm512_cmpgt_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)       return simde_mm512_cmpgt_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)      return simde_mm512_cmpgt_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }
      
      /// Compare values as constexpr, if possible                            
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return bool/bitmask                                              
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto GreaterConstexpr(const auto& lhs, const auto& rhs) noexcept {
         // Will always return a std::array<bool>                       
         constexpr auto S = OverlapCounts<decltype(lhs), decltype(rhs)>();

         return AttemptBinary<0, ::std::array<bool, S>>(
            lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> bool {
               return l > r;
            }
         );
      }
   
      /// Compare values as a register, if possible                           
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return bool/bitmask/register                                     
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Greater(const auto& lhs, const auto& rhs) noexcept {
         // Will return either a std::array<bool>, or a masked register,
         // depending whether SIMD operation is supported or not        
         constexpr auto S = OverlapCounts<decltype(lhs), decltype(rhs)>();
         using OUT = Conditional<CT::SIMD<FORCE_OUT>, FORCE_OUT, ::std::array<bool, S>>;

         return AttemptBinary<0, OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Comparing for greater (SIMD) as ", NameOf<R>());
               return GreaterSIMD(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> bool {
               LANGULUS_SIMD_VERBOSE("Comparing for greater (Fallback) ", l, " > ", r, " (", NameOf<E>(), ")");
               return l > r;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   /// Compare numbers for great, force output to desired place               
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in 'out'. Use Inner::Greater if you    
   ///      don't want this.                                                  
   template<class LHS, class RHS, CT::NotSemantic OUT> LANGULUS(INLINED)
   constexpr void Greater(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
      IF_CONSTEXPR() {
         Store(Inner::GreaterConstexpr<OUT>(DesemCast(lhs), DesemCast(rhs)), out);
      }
      else {
         Store(Inner::Greater<OUT>(DesemCast(lhs), DesemCast(rhs)), out);
      }
   }

   /// Compare numbers for great                                              
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in an instance of 'OUT'. Use           
   ///      Inner::Greater if you don't want this.                            
   template<class LHS, class RHS, CT::NotSemantic OUT = Bitmask<OverlapCounts<LHS, RHS>()>>
   LANGULUS(INLINED)
   constexpr OUT Greater(const LHS& lhs, const RHS& rhs) noexcept {
      OUT out;
      Greater(DesemCast(lhs), DesemCast(rhs), out);
      return out;
   }

} // namespace Langulus::SIMD