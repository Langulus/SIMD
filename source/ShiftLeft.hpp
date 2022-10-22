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
   LANGULUS(ALWAYSINLINE) constexpr auto ShiftLeftInner(CT::NotSupported auto, CT::NotSupported auto) noexcept {
      return CT::Inner::NotSupported{};
   }

   /// Shift two arrays left using SIMD (shifting in zeroes)                  
   ///   @attention this differs from C++'s undefined behavior when shifting  
   ///      by less than zero, or by a number larger than the bitcount.       
   ///      the SIMD operations define this behavior very well, by just       
   ///      defaulting to zero. It is our responsibility to keep this         
   ///      behavior consistent across C++ and SIMD, so the fallback routine  
   ///      has additional overhead for checking the rhs range and zeroing.   
   ///   @tparam T - the type of the array element                            
   ///   @tparam S - the size of the array                                    
   ///   @tparam REGISTER - type of register we're operating with             
   ///   @param lhs - the left-hand-side array                                
   ///   @param rhs - the right-hand-side array                               
   ///   @return the shifted elements as a register                           
   template<class T, Count S, CT::TSIMD REGISTER>
   LANGULUS(ALWAYSINLINE) auto ShiftLeftInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
      static_assert(CT::IntegerX<Decay<T>>, "Can only shift integers");

      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::Integer8<T>) {
               const auto zero = simde_mm_setzero_si128();
               auto lhs1 = simde_mm_unpacklo_epi8(lhs, zero);
               auto rhs1 = simde_mm_unpacklo_epi8(rhs, zero);
               auto lhs2 = simde_mm_unpackhi_epi8(lhs, zero);
               auto rhs2 = simde_mm_unpackhi_epi8(rhs, zero);
               #if LANGULUS_SIMD(512BIT)
                  // Optimal                                            
                  lhs1 = simde_mm_sllv_epi16(lhs1, rhs1);
                  lhs2 = simde_mm_sllv_epi16(lhs2, rhs2);
                  return lgls_pack_epi16(lhs1, lhs2);
               #elif LANGULUS_SIMD(256BIT)
                  // Not optimal, must be unpacked once more for AVX2   
                  auto lhs32_1 = simde_mm_unpacklo_epi16(lhs1, zero);
                  auto rhs32_1 = simde_mm_unpacklo_epi16(rhs1, zero);
                  auto lhs32_2 = simde_mm_unpackhi_epi16(lhs1, zero);
                  auto rhs32_2 = simde_mm_unpackhi_epi16(rhs1, zero);

                  lhs32_1 = simde_mm_sllv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm_sllv_epi32(lhs32_2, rhs32_2);
                  lhs1 = lgls_pack_epi32(lhs32_1, lhs32_2);

                  lhs32_1 = simde_mm_unpacklo_epi16(lhs2, zero);
                  rhs32_1 = simde_mm_unpacklo_epi16(rhs2, zero);
                  lhs32_2 = simde_mm_unpackhi_epi16(lhs2, zero);
                  rhs32_2 = simde_mm_unpackhi_epi16(rhs2, zero);

                  lhs32_1 = simde_mm_sllv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm_sllv_epi32(lhs32_2, rhs32_2);
                  lhs2 = lgls_pack_epi32(lhs32_1, lhs32_2);

                  return lgls_pack_epi16(lhs1, lhs2);
               #else
                  return CT::Inner::NotSupported{}; //TODO
               #endif
            }
            else if constexpr (CT::Integer16<T>) {
               const auto zero = simde_mm_setzero_si128();
               #if LANGULUS_SIMD(512BIT)
                  // Optimal                                            
                  return simde_mm_sllv_epi16(lhs, rhs);
               #elif LANGULUS_SIMD(256BIT)
                  // Not optimal, must be unpacked for AVX2             
                  auto lhs32_1 = simde_mm_unpacklo_epi16(lhs, zero);
                  auto rhs32_1 = simde_mm_unpacklo_epi16(rhs, zero);
                  auto lhs32_2 = simde_mm_unpackhi_epi16(lhs, zero);
                  auto rhs32_2 = simde_mm_unpackhi_epi16(rhs, zero);

                  lhs32_1 = simde_mm_sllv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm_sllv_epi32(lhs32_2, rhs32_2);
                  return lgls_pack_epi32(lhs32_1, lhs32_2);
               #else
                  return CT::Inner::NotSupported{}; //TODO
               #endif
            }
            else if constexpr (CT::Integer32<T>) {
               #if LANGULUS_SIMD(256BIT)
                  return simde_mm_sllv_epi32(lhs, rhs);
               #else
                  return CT::Inner::NotSupported {}; //TODO
               #endif
            }
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(256BIT)
                  return simde_mm_sllv_epi64(lhs, rhs);
               #else
                  return CT::Inner::NotSupported {}; //TODO
               #endif
            }
            else LANGULUS_ERROR("Unsupported type for SIMD::ShiftLeftInner of 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::Integer8<T>) {
               const auto zero = simde_mm256_setzero_si256();
               auto lhs1 = simde_mm256_cvtepu8_epi16(simde_mm256_extracti128_si256(lhs, 0));
               auto lhs2 = simde_mm256_cvtepu8_epi16(simde_mm256_extracti128_si256(lhs, 1));
               auto rhs1 = simde_mm256_cvtepu8_epi16(simde_mm256_extracti128_si256(rhs, 0));
               auto rhs2 = simde_mm256_cvtepu8_epi16(simde_mm256_extracti128_si256(rhs, 1));
               #if LANGULUS_SIMD(512BIT)
                  // Optimal                                            
                  lhs1 = simde_mm256_sllv_epi16(lhs1, rhs1);
                  lhs2 = simde_mm256_sllv_epi16(lhs2, rhs2);
                  return lgls_pack_epi16(lhs1, lhs2);
               #else
                  // Not optimal, must be unpacked once more for AVX2   
                  auto lhs32_1 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(lhs1, 0));
                  auto lhs32_2 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(lhs1, 1));
                  auto rhs32_1 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(rhs1, 0));
                  auto rhs32_2 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(rhs1, 1));

                  lhs32_1 = simde_mm256_sllv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm256_sllv_epi32(lhs32_2, rhs32_2);
                  lhs1 = lgls_pack_epi32(lhs32_1, lhs32_2);

                  lhs32_1 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(lhs2, 0));
                  lhs32_2 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(lhs2, 1));
                  rhs32_1 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(rhs2, 0));
                  rhs32_2 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(rhs2, 1));

                  lhs32_1 = simde_mm256_sllv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm256_sllv_epi32(lhs32_2, rhs32_2);
                  lhs2 = lgls_pack_epi32(lhs32_1, lhs32_2);

                  return lgls_pack_epi16(lhs1, lhs2);
               #endif
            }
            else if constexpr (CT::Integer16<T>) {
               const auto zero = simde_mm256_setzero_si256();
               #if LANGULUS_SIMD(512BIT)
                  // Optimal                                            
                  return simde_mm256_sllv_epi16(lhs, rhs);
               #else
                  // Not optimal, must be unpacked for AVX2             
                  auto lhs32_1 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(lhs, 0));
                  auto lhs32_2 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(lhs, 1));
                  auto rhs32_1 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(rhs, 0));
                  auto rhs32_2 = simde_mm256_cvtepu16_epi32(simde_mm256_extracti128_si256(rhs, 1));

                  lhs32_1 = simde_mm256_sllv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm256_sllv_epi32(lhs32_2, rhs32_2);
                  return lgls_pack_epi32(lhs32_1, lhs32_2);
               #endif
            }
            else if constexpr (CT::Integer32<T>)
               return simde_mm256_sllv_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)
               return simde_mm256_sllv_epi64(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::ShiftLeftInner of 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::Integer8<T>) {
               const auto zero = simde_mm512_setzero_si512();
               auto lhs1 = simde_mm512_unpacklo_epi8(lhs, zero);
               auto rhs1 = simde_mm512_unpacklo_epi8(rhs, zero);
               auto lhs2 = simde_mm512_unpackhi_epi8(lhs, zero);
               auto rhs2 = simde_mm512_unpackhi_epi8(rhs, zero);

               lhs1 = simde_mm512_sllv_epi16(lhs1, rhs1);
               lhs2 = simde_mm512_sllv_epi16(lhs2, rhs2);
               return lgls_pack_epi16(lhs1, lhs2);
            }
            else if constexpr (CT::Integer16<T>)
               return simde_mm512_sllv_epi16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)
               return simde_mm512_sllv_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)
               return simde_mm512_sllv_epi64(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::InnerShiftLeft of 64-byte package");
         }
         else
      #endif
         LANGULUS_ERROR("Unsupported type for SIMD::ShiftLeftInner");
   }

   ///   @attention this differs from C++'s undefined behavior when shifting  
   ///      by less than zero, or by a number larger than the bitcount.       
   ///      the SIMD operations define this behavior very well, by just       
   ///      defaulting to zero. It is our responsibility to keep this         
   ///      behavior consistent across C++ and SIMD, so the fallback routine  
   ///      has additional overhead for checking the rhs range and zeroing.   
   template<class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) auto ShiftLeft(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      static_assert(CT::IntegerX<Decay<LHS>, Decay<RHS>>, "Can only shift integers");
      using REGISTER = CT::Register<LHS, RHS>;
      using LOSSLESS = Lossless<LHS, RHS>;
      constexpr auto S = OverlapCount<LHS, RHS>();

      return AttemptSIMD<0, REGISTER, LOSSLESS>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return ShiftLeftInner<LOSSLESS, S>(lhs, rhs);
         },
         [](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept -> LOSSLESS {
            // Well defined condition in SIMD calls, that is otherwise  
            // undefined behavior by C++ standard. Practically errorsome
            // only on clang, but still, better safe than sorry			
            return rhs < LOSSLESS {sizeof(LOSSLESS) * 8} && rhs >= 0
               ? lhs << rhs : 0;
         }
      );
   }

   ///   @attention this differs from C++'s undefined behavior when shifting  
   ///      by less than zero, or by a number larger than the bitcount.       
   ///      the SIMD operations define this behavior very well, by just       
   ///      defaulting to zero. It is our responsibility to keep this         
   ///      behavior consistent across C++ and SIMD, so the fallback routine  
   ///      has additional overhead for checking the rhs range and zeroing.   
   template<class LHS, class RHS, class OUT>
   LANGULUS(ALWAYSINLINE) void ShiftLeft(const LHS& lhs, const RHS& rhs, OUT& output) {
      static_assert(CT::IntegerX<Decay<LHS>, Decay<RHS>>, "Can only shift integers");
      GeneralStore(ShiftLeft<LHS, RHS>(lhs, rhs), output);
   }

   ///   @attention this differs from C++'s undefined behavior when shifting  
   ///      by less than zero, or by a number larger than the bitcount.       
   ///      the SIMD operations define this behavior very well, by just       
   ///      defaulting to zero. It is our responsibility to keep this         
   ///      behavior consistent across C++ and SIMD, so the fallback routine  
   ///      has additional overhead for checking the rhs range and zeroing.   
   template<CT::Vector WRAPPER, class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) WRAPPER ShiftLeftWrap(const LHS& lhs, const RHS& rhs) {
      static_assert(CT::IntegerX<Decay<LHS>, Decay<RHS>>, "Can only shift integers");
      WRAPPER result;
      ShiftLeft<LHS, RHS>(lhs, rhs, result.mArray);
      return result;
   }

} // namespace Langulus::SIMD