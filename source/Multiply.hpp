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
#include "IgnoreWarningsPush.inl"

namespace Langulus::SIMD
{

   template<class, Count>
   LANGULUS(ALWAYSINLINE) constexpr auto MultiplyInner(CT::NotSupported auto, CT::NotSupported auto) noexcept {
      return CT::Inner::NotSupported{};
   }

   /// Multiply two arrays using SIMD                                         
   ///   @tparam T - the type of the array element                            
   ///   @tparam S - the size of the array                                    
   ///   @tparam REGISTER - type of register we're operating with             
   ///   @param lhs - the left-hand-side array                                
   ///   @param rhs - the right-hand-side array                               
   ///   @return the multiplied elements as a register                        
   template<class T, Count S, CT::TSIMD REGISTER>
   LANGULUS(ALWAYSINLINE) auto MultiplyInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::Integer8<T>) {
               // https://stackoverflow.com/questions/8193601              
               simde__m128i zero = simde_mm_setzero_si128();
               simde__m128i Alo = simde_mm_cvtepu8_epi16(lhs);
               simde__m128i Ahi = simde_mm_unpackhi_epi8(lhs, zero);
               simde__m128i Blo = simde_mm_cvtepu8_epi16(rhs);
               simde__m128i Bhi = simde_mm_unpackhi_epi8(rhs, zero);
               simde__m128i Clo = simde_mm_mullo_epi16(Alo, Blo);
               simde__m128i Chi = simde_mm_mullo_epi16(Ahi, Bhi);
               return lgls_pack_epi16(Clo, Chi);
            }
            else if constexpr (CT::Integer16<T>)
               return simde_mm_mullo_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)
               return simde_mm_mullo_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm_mullo_epi64(lhs, rhs);
               #else
                  return CT::Inner::NotSupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)
               return simde_mm_mul_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm_mul_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::MultiplyInner of 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::Integer8<T>) {
               simde__m256i Alo = simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(lhs));
               simde__m256i Ahi = simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(_mm_halfflip(lhs)));
               simde__m256i Blo = simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(rhs));
               simde__m256i Bhi = simde_mm256_cvtepu8_epi16(_mm256_castsi256_si128(_mm_halfflip(rhs)));
               simde__m256i Clo = simde_mm256_mullo_epi16(Alo, Blo);
               simde__m256i Chi = simde_mm256_mullo_epi16(Ahi, Bhi);
               return lgls_pack_epi16(Clo, Chi);
            }
            else if constexpr (CT::Integer16<T>)
               return simde_mm256_mullo_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)
               return simde_mm256_mullo_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm256_mullo_epi64(lhs, rhs);
               #else
                  return CT::Inner::NotSupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)
               return simde_mm256_mul_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm256_mul_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::MultiplyInner of 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::Integer8<T>) {
               auto hiLHS = simde_mm512_unpackhi_epi8(lhs, simde_mm512_setzero_si512());
               auto hiRHS = simde_mm512_unpackhi_epi8(rhs, simde_mm512_setzero_si512());
               hiLHS = simde_mm512_mullo_epi16(hiLHS, hiRHS);

               auto loLHS = simde_mm512_unpacklo_epi8(lhs, simde_mm512_setzero_si512());
               auto loRHS = simde_mm256_unpacklo_epi8(rhs, simde_mm512_setzero_si512());
               loLHS = simde_mm512_mullo_epi16(loLHS, loRHS);

               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm512_packs_epi16(loLHS, hiLHS);
               else
                  return simde_mm512_packus_epi16(loLHS, hiLHS);
            }
            else if constexpr (CT::Integer16<T>)
               return simde_mm512_mullo_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)
               return simde_mm512_mullo_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return _mm512_mullo_epi64(lhs, rhs);
               #else
                  return CT::Inner::NotSupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)
               return simde_mm512_mul_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm512_mul_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::MultiplyInner of 64-byte package");
         }
         else
      #endif
         LANGULUS_ERROR("Unsupported type for SIMD::MultiplyInner");
   }

   ///                                                                        
   template<class LHS, class RHS, class OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(ALWAYSINLINE) auto Multiply(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using DOUT = Decay<OUT>;
      using REGISTER = CT::Register<LHS, RHS, DOUT>;
      constexpr auto S = OverlapCount<LHS, RHS>();

      return AttemptSIMD<0, REGISTER, DOUT>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return MultiplyInner<DOUT, S>(lhs, rhs);
         },
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            return lhs * rhs;
         }
      );
   }

   ///                                                                        
   template<class LHS, class RHS, class OUT>
   LANGULUS(ALWAYSINLINE) void Multiply(const LHS& lhs, const RHS& rhs, OUT& output) noexcept {
      GeneralStore(Multiply<LHS, RHS, OUT>(lhs, rhs), output);
   }

   ///                                                                        
   template<CT::Vector WRAPPER, class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) WRAPPER MultiplyWrap(const LHS& lhs, const RHS& rhs) noexcept {
      WRAPPER result;
      Multiply<LHS, RHS>(lhs, rhs, result.mArray);
      return result;
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
