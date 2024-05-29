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
#include "IgnoreWarningsPush.inl"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<CT::Decayed, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported MultiplySIMD(const T&, const T&) noexcept {
         return {};
      }

      /// Multiply two arrays using SIMD                                      
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - type of register we're operating with          
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the multiplied elements as a register                     
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto MultiplySIMD(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
         #if LANGULUS_SIMD(128BIT)
            if constexpr (CT::SIMD128<REGISTER>) {
               if constexpr (CT::Integer8<T>) {
                  // https://stackoverflow.com/questions/8193601        
                  /*simde__m128i zero = simde_mm_setzero_si128();
                  simde__m128i Alo = simde_mm_cvtepu8_epi16(lhs);
                  simde__m128i Ahi = simde_mm_unpackhi_epi8(lhs, zero);
                  simde__m128i Blo = simde_mm_cvtepu8_epi16(rhs);
                  simde__m128i Bhi = simde_mm_unpackhi_epi8(rhs, zero);
                  simde__m128i Clo = simde_mm_mullo_epi16(Alo, Blo);
                  simde__m128i Chi = simde_mm_mullo_epi16(Ahi, Bhi);
                  return lgls_pack_epi16(Clo, Chi);*/
                  return Unsupported {};
               }
               else if constexpr (CT::Integer16<T>)
                  return simde_mm_mullo_epi16(lhs, rhs);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm_mullo_epi32(lhs, rhs);
               else if constexpr (CT::Integer64<T>) {
                  #if LANGULUS_SIMD(AVX512)
                     return _mm_mullo_epi64(lhs, rhs);
                  #else
                     return Unsupported{};
                  #endif
               }
               else if constexpr (CT::Float<T>)
                  return simde_mm_mul_ps(lhs, rhs);
               else if constexpr (CT::Double<T>)
                  return simde_mm_mul_pd(lhs, rhs);
               else
                  LANGULUS_ERROR("Unsupported type for 16-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(256BIT)
            if constexpr (CT::SIMD256<REGISTER>) {
               if constexpr (CT::Integer8<T>) {
                  /*simde__m256i Alo = simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(lhs));
                  simde__m256i Ahi = simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(_mm_halfflip(lhs)));
                  simde__m256i Blo = simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(rhs));
                  simde__m256i Bhi = simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(_mm_halfflip(rhs)));
                  simde__m256i Clo = simde_mm256_mullo_epi16(Alo, Blo);
                  simde__m256i Chi = simde_mm256_mullo_epi16(Ahi, Bhi);
                  return lgls_pack_epi16(Clo, Chi);*/
                  return Unsupported {};
               }
               else if constexpr (CT::Integer16<T>)
                  return simde_mm256_mullo_epi16(lhs, rhs);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm256_mullo_epi32(lhs, rhs);
               else if constexpr (CT::Integer64<T>) {
                  #if LANGULUS_SIMD(AVX512)
                     return _mm256_mullo_epi64(lhs, rhs);
                  #else
                     return Unsupported{};
                  #endif
               }
               else if constexpr (CT::Float<T>)
                  return simde_mm256_mul_ps(lhs, rhs);
               else if constexpr (CT::Double<T>)
                  return simde_mm256_mul_pd(lhs, rhs);
               else
                  LANGULUS_ERROR("Unsupported type for 32-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(512BIT)
            if constexpr (CT::SIMD512<REGISTER>) {
               if constexpr (CT::Integer8<T>) {
                  /*auto hiLHS = simde_mm512_unpackhi_epi8(lhs, simde_mm512_setzero_si512());
                  auto hiRHS = simde_mm512_unpackhi_epi8(rhs, simde_mm512_setzero_si512());
                  hiLHS = simde_mm512_mullo_epi16(hiLHS, hiRHS);

                  auto loLHS = simde_mm512_unpacklo_epi8(lhs, simde_mm512_setzero_si512());
                  auto loRHS = simde_mm256_unpacklo_epi8(rhs, simde_mm512_setzero_si512());
                  loLHS = simde_mm512_mullo_epi16(loLHS, loRHS);

                  if constexpr (CT::SignedInteger8<T>)
                     return simde_mm512_packs_epi16(loLHS, hiLHS);
                  else
                     return simde_mm512_packus_epi16(loLHS, hiLHS);*/
                  return Unsupported {};
               }
               else if constexpr (CT::Integer16<T>)
                  return simde_mm512_mullo_epi16(lhs, rhs);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm512_mullo_epi32(lhs, rhs);
               else if constexpr (CT::Integer64<T>) {
                  #if LANGULUS_SIMD(AVX512)
                     return _mm512_mullo_epi64(lhs, rhs);
                  #else
                     return Unsupported{};
                  #endif
               }
               else if constexpr (CT::Float<T>)
                  return simde_mm512_mul_ps(lhs, rhs);
               else if constexpr (CT::Double<T>)
                  return simde_mm512_mul_pd(lhs, rhs);
               else
                  LANGULUS_ERROR("Unsupported type for 64-byte package");
            }
            else
         #endif
            LANGULUS_ERROR("Unsupported type");
      }

      /// Multiply numbers and return a register, if possible                 
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto MultiplyConstexpr(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         using DOUT = TypeOf<SIMD::LosslessArray<OUT, OUT>>;

         return Evaluate2<0, Unsupported, OUT>(
            lhsOrig, rhsOrig, nullptr,
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
               if constexpr (CT::Same<DOUT, uint8_t>) {
                  // 8-bit unsigned multiplication with saturation      
                  const unsigned temp = lhs * rhs;
                  return temp > 255 ? 255 : temp;
               }
               else if constexpr (CT::Same<DOUT, int8_t>) {
                  // 8-bit signed multiplication with saturation        
                  const signed temp = lhs * rhs;
                  return temp > 127 ? 127 : (temp < -128 ? -128 : temp);
               }
               else return lhs * rhs;
            }
         );
      }

      /// Multiply numbers and return a register, if possible                 
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto MultiplyDynamic(const auto& lhsOrig, const auto& rhsOrig) noexcept {
         using DOUT = TypeOf<SIMD::LosslessArray<OUT, OUT>>;
         using REGISTER = Register<decltype(lhsOrig), decltype(rhsOrig), OUT>;

         return Evaluate2<0, REGISTER, OUT>(
            lhsOrig, rhsOrig,
            [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
               LANGULUS_SIMD_VERBOSE("Multiplying (SIMD)");
               return MultiplySIMD<DOUT>(lhs, rhs);
            },
            [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
               LANGULUS_SIMD_VERBOSE("Multiplying (Fallback)");
               if constexpr (CT::Same<DOUT, uint8_t>) {
                  // 8-bit unsigned multiplication with saturation      
                  const unsigned temp = lhs * rhs;
                  return temp > 255 ? 255 : temp;
               }
               else if constexpr (CT::Same<DOUT, int8_t>) {
                  // 8-bit signed multiplication with saturation        
                  const signed temp = lhs * rhs;
                  return temp > 127 ? 127 : (temp < -128 ? -128 : temp);
               }
               else return lhs * rhs;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner


   /// Multiply numbers, and force output to desired place                    
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<class LHS, class RHS, CT::NotSemantic OUT> LANGULUS(INLINED)
   constexpr void Multiply(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
      IF_CONSTEXPR() {
      StoreConstexpr(
         Inner::MultiplyConstexpr<OUT>(DesemCast(lhs), DesemCast(rhs)), out);
      }
      else Store(
         Inner::MultiplyDynamic<OUT>(DesemCast(lhs), DesemCast(rhs)), out);
   }
      
   /// Multiply numbers                                                       
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<class LHS, class RHS, CT::NotSemantic OUT = LosslessArray<LHS, RHS>>
   LANGULUS(INLINED)
   constexpr auto Multiply(const LHS& lhs, const RHS& rhs) noexcept {
      OUT out;
      Multiply(DesemCast(lhs), DesemCast(rhs), out);

      if constexpr (CT::Similar<LHS, RHS> or CT::DerivedFrom<LHS, RHS>)
         return LHS {out};
      else if constexpr (CT::DerivedFrom<RHS, LHS>)
         return RHS {out};
      else
         return out;
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
