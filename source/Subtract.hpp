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
#include "IgnoreWarningsPush.inl"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<CT::Decayed, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported Subtract(const T&, const T&) noexcept {
         return {};
      }

      /// Subtract two arrays using SIMD                                      
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - type of register we're operating with          
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the subtracted elements as a register                     
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto Subtract(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
         #if LANGULUS_SIMD(128BIT)
            if constexpr (CT::SIMD128<REGISTER>) {
               if constexpr (CT::Integer8<T>)
                  return simde_mm_sub_epi8(lhs, rhs);
               else if constexpr (CT::Integer16<T>)
                  return simde_mm_sub_epi16(lhs, rhs);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm_sub_epi32(lhs, rhs);
               else if constexpr (CT::Integer64<T>)
                  return simde_mm_sub_epi64(lhs, rhs);
               else if constexpr (CT::Float<T>)
                  return simde_mm_sub_ps(lhs, rhs);
               else if constexpr (CT::Double<T>)
                  return simde_mm_sub_pd(lhs, rhs);
               else
                  LANGULUS_ERROR("Unsupported type for 16-byte package");
            }
            else
         #endif
         
         #if LANGULUS_SIMD(256BIT)
            if constexpr (CT::SIMD256<REGISTER>) {
               if constexpr (CT::Integer8<T>)
                  return simde_mm256_sub_epi8(lhs, rhs);
               else if constexpr (CT::Integer16<T>)
                  return simde_mm256_sub_epi16(lhs, rhs);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm256_sub_epi32(lhs, rhs);
               else if constexpr (CT::Integer64<T>)
                  return simde_mm256_sub_epi64(lhs, rhs);
               else if constexpr (CT::Float<T>)
                  return simde_mm256_sub_ps(lhs, rhs);
               else if constexpr (CT::Double<T>)
                  return simde_mm256_sub_pd(lhs, rhs);
               else
                  LANGULUS_ERROR("Unsupported type for 32-byte package");
            }
            else
         #endif
         
         #if LANGULUS_SIMD(512BIT)
            if constexpr (CT::SIMD512<REGISTER>) {
               if constexpr (CT::Integer8<T>)
                  return simde_mm512_sub_epi8(lhs, rhs);
               else if constexpr (CT::Integer16<T>)
                  return simde_mm512_sub_epi16(lhs, rhs);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm512_sub_epi32(lhs, rhs);
               else if constexpr (CT::Integer64<T>)
                  return simde_mm512_sub_epi64(lhs, rhs);
               else if constexpr (CT::Float<T>)
                  return simde_mm512_sub_ps(lhs, rhs);
               else if constexpr (CT::Double<T>)
                  return simde_mm512_sub_pd(lhs, rhs);
               else
                  LANGULUS_ERROR("Unsupported type for 64-byte package");
            }
            else
         #endif
            LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Subtract numbers                                                       
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   constexpr auto SubtractConstexpr(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using DOUT = Decay<TypeOf<OUT>>;

      return Inner::Evaluate2<0, Unsupported, OUT>(
         lhsOrig, rhsOrig, nullptr,
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            return lhs - rhs;
         }
      );
   }

   /// Subtract numbers                                                       
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   auto SubtractDynamic(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using DOUT = Decay<TypeOf<OUT>>;
      using REGISTER = Inner::Register<LHS, RHS, OUT>;

      return Inner::Evaluate2<0, REGISTER, OUT>(
         lhsOrig, rhsOrig,
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return Inner::Subtract<DOUT>(lhs, rhs);
         },
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            return lhs - rhs;
         }
      );
   }

   /// Subtract numbers, and force output to desired place                    
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT> LANGULUS(INLINED)
   constexpr void Subtract(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
      IF_CONSTEXPR() {
         StoreConstexpr(SubtractConstexpr<LHS, RHS, OUT>(lhs, rhs), out);
      }
      else Store(SubtractDynamic<LHS, RHS, OUT>(lhs, rhs), out);
   }

   /// Subtract numbers                                                       
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = std::array<Lossless<Decay<TypeOf<LHS>>, Decay<TypeOf<RHS>>>, OverlapCounts<LHS, RHS>()>>
   LANGULUS(INLINED)
   constexpr OUT Subtract(const LHS& lhs, const RHS& rhs) noexcept {
      OUT out;
      Subtract(lhs, rhs, out);
      return out;
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
