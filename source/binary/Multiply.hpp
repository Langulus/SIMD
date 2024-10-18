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
      constexpr Unsupported MultiplySIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Multiply two registers                                              
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      auto MultiplySIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         (void)lhs; (void)rhs;

         if constexpr (CT::SIMD128<R>) {
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
            else if constexpr (CT::Integer16<T>)      return R {simde_mm_mullo_epi16(lhs, rhs)};
            else if constexpr (CT::Integer32<T>)      return R {simde_mm_mullo_epi32(lhs, rhs)};
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return R {_mm_mullo_epi64(lhs, rhs)};
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)          return R {simde_mm_mul_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)         return R {simde_mm_mul_pd(lhs, rhs)};
            else static_assert(false, "Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
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
            else if constexpr (CT::Integer16<T>)      return R {simde_mm256_mullo_epi16(lhs, rhs)};
            else if constexpr (CT::Integer32<T>)      return R {simde_mm256_mullo_epi32(lhs, rhs)};
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return R {_mm256_mullo_epi64(lhs, rhs)};
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)          return R {simde_mm256_mul_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)         return R {simde_mm256_mul_pd(lhs, rhs)};
            else static_assert(false, "Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
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
            else if constexpr (CT::Integer16<T>)      return R {simde_mm512_mullo_epi16(lhs, rhs)};
            else if constexpr (CT::Integer32<T>)      return R {simde_mm512_mullo_epi32(lhs, rhs)};
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(AVX512)
                  return R {_mm512_mullo_epi64(lhs, rhs)};
               #else
                  return Unsupported{};
               #endif
            }
            else if constexpr (CT::Float<T>)          return R {simde_mm512_mul_ps(lhs, rhs)};
            else if constexpr (CT::Double<T>)         return R {simde_mm512_mul_pd(lhs, rhs)};
            else static_assert(false, "Unsupported type for 64-byte package");
         }
         else static_assert(false, "Unsupported type");
      }
      
      /// Get product of values as constexpr, if possible                     
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the product scalar/vector                                 
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto MultiplyConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> E {
               if constexpr (CT::Same<E, uint8_t>) {
                  // 8-bit unsigned multiplication with saturation      
                  const unsigned temp = l * r;
                  return temp > 255 ? 255 : temp;
               }
               else if constexpr (CT::Same<E, int8_t>) {
                  // 8-bit signed multiplication with saturation        
                  const signed temp = l * r;
                  return temp > 127 ? 127 : (temp < -128 ? -128 : temp);
               }
               else return l * r;
            }
         );
      }
   
      /// Get product values as a register, if possible                       
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the product scalar/vector/register                        
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto Multiply(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               LANGULUS_SIMD_VERBOSE("Multiplying (SIMD) as ", NameOf<REGISTER>());
               return MultiplySIMD(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> E {
               LANGULUS_SIMD_VERBOSE("Multiplying (Fallback) ", l, " * ", r, " (", NameOf<E>(), ")");
               if constexpr (CT::Same<E, uint8_t>) {
                  // 8-bit unsigned multiplication with saturation      
                  const unsigned temp = l * r;
                  return temp > 255 ? 255 : temp;
               }
               else if constexpr (CT::Same<E, int8_t>) {
                  // 8-bit signed multiplication with saturation        
                  const signed temp = l * r;
                  return temp > 127 ? 127 : (temp < -128 ? -128 : temp);
               }
               else return l * r;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(Multiply)

} // namespace Langulus::SIMD
