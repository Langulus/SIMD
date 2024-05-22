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


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<CT::Decayed, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported ShiftLeft(const T&, const T&) noexcept {
         return {};
      }

      /// Shift two arrays left using SIMD (shifting in zeroes)               
      ///   @attention this differs from C++'s undefined behavior when shifting
      ///      by less than zero, or by a number larger than the bitcount.    
      ///      the SIMD operations define this behavior very well, by just    
      ///      defaulting to zero. It is our responsibility to keep this      
      ///      behavior consistent across C++ and SIMD, so the fallback routine
      ///      has additional overhead for checking the rhs range and zeroing.
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - type of register we're operating with          
      ///   @param lhs - the left-hand-side array                             
      ///   @param rhs - the right-hand-side array                            
      ///   @return the shifted elements as a register                        
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto ShiftLeft(UNUSED() const REGISTER& lhs, UNUSED() const REGISTER& rhs) noexcept {
         static_assert(CT::IntegerX<Decay<T>>, "Can only shift integers");

         #if LANGULUS_SIMD(128BIT)
            if constexpr (CT::SIMD128<REGISTER>) {
               if constexpr (CT::Integer8<T>) {
                  #if LANGULUS_SIMD(512BIT)
                     // Optimal                                         
                     const auto zero = simde_mm_setzero_si128();
                     auto lhs1 = simde_mm_unpacklo_epi8(lhs, zero);
                     auto rhs1 = simde_mm_unpacklo_epi8(rhs, zero);
                     auto lhs2 = simde_mm_unpackhi_epi8(lhs, zero);
                     auto rhs2 = simde_mm_unpackhi_epi8(rhs, zero);
                     lhs1 = simde_mm_sllv_epi16(lhs1, rhs1);
                     lhs2 = simde_mm_sllv_epi16(lhs2, rhs2);
                     return lgls_pack_epi16(lhs1, lhs2);
                  #elif LANGULUS_SIMD(256BIT)
                     // Not optimal, must be unpacked once more for AVX2
                     const auto zero = simde_mm_setzero_si128();
                     auto lhs1 = simde_mm_unpacklo_epi8(lhs, zero);
                     auto rhs1 = simde_mm_unpacklo_epi8(rhs, zero);
                     auto lhs2 = simde_mm_unpackhi_epi8(lhs, zero);
                     auto rhs2 = simde_mm_unpackhi_epi8(rhs, zero);
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
                     (void)lhs;
                     (void)rhs;
                     return Unsupported{}; //TODO
                  #endif
               }
               else if constexpr (CT::Integer16<T>) {
                  #if LANGULUS_SIMD(512BIT)
                     // Optimal                                         
                     return simde_mm_sllv_epi16(lhs, rhs);
                  #elif LANGULUS_SIMD(256BIT)
                     // Not optimal, must be unpacked for AVX2          
                     const auto zero = simde_mm_setzero_si128();
                     auto lhs32_1 = simde_mm_unpacklo_epi16(lhs, zero);
                     auto rhs32_1 = simde_mm_unpacklo_epi16(rhs, zero);
                     auto lhs32_2 = simde_mm_unpackhi_epi16(lhs, zero);
                     auto rhs32_2 = simde_mm_unpackhi_epi16(rhs, zero);

                     lhs32_1 = simde_mm_sllv_epi32(lhs32_1, rhs32_1);
                     lhs32_2 = simde_mm_sllv_epi32(lhs32_2, rhs32_2);
                     return lgls_pack_epi32(lhs32_1, lhs32_2);
                  #else
                     (void)lhs;
                     (void)rhs;
                     return Unsupported{}; //TODO
                  #endif
               }
               else if constexpr (CT::Integer32<T>) {
                  #if LANGULUS_SIMD(256BIT)
                     return simde_mm_sllv_epi32(lhs, rhs);
                  #else
                     (void)lhs;
                     (void)rhs;
                     return Unsupported {}; //TODO
                  #endif
               }
               else if constexpr (CT::Integer64<T>) {
                  #if LANGULUS_SIMD(256BIT)
                     return simde_mm_sllv_epi64(lhs, rhs);
                  #else
                     (void)lhs;
                     (void)rhs;
                     return Unsupported {}; //TODO
                  #endif
               }
               else LANGULUS_ERROR("Unsupported type for 16-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(256BIT)
            if constexpr (CT::SIMD256<REGISTER>) {
               if constexpr (CT::Integer8<T>) {
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
                  LANGULUS_ERROR("Unsupported type for 32-byte package");
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
                  LANGULUS_ERROR("Unsupported type for 64-byte package");
            }
            else
         #endif
            LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Shift bits left                                                        
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   ///   @attention this differs from C++'s undefined behavior when shifting  
   ///      by less than zero, or by a number larger than the bitcount.       
   ///      the SIMD operations define this behavior very well, by just       
   ///      defaulting to zero. It is our responsibility to keep this         
   ///      behavior consistent across C++ and SIMD, so the fallback routine  
   ///      has additional overhead for checking the rhs range and zeroing.   
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   constexpr auto ShiftLeftConstexpr(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      static_assert(CT::IntegerX<Decay<TypeOf<LHS>>, Decay<TypeOf<RHS>>>,
         "Can only shift integers");

      using DOUT = Decay<TypeOf<OUT>>;

      return Inner::Evaluate2<0, Unsupported, OUT>(
         lhsOrig, rhsOrig, nullptr,
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            // Well defined condition in SIMD calls, that is otherwise  
            // undefined behavior by C++ standard                       
            return rhs < DOUT {sizeof(DOUT) * 8} and rhs >= 0
               ? lhs << rhs : 0;
         }
      );
   }
   
   /// Shift bits left                                                        
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (lossless by default)         
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   ///   @attention this differs from C++'s undefined behavior when shifting  
   ///      by less than zero, or by a number larger than the bitcount.       
   ///      the SIMD operations define this behavior very well, by just       
   ///      defaulting to zero. It is our responsibility to keep this         
   ///      behavior consistent across C++ and SIMD, so the fallback routine  
   ///      has additional overhead for checking the rhs range and zeroing.   
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(INLINED)
   auto ShiftLeftDynamic(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      static_assert(CT::IntegerX<Decay<TypeOf<LHS>>, Decay<TypeOf<RHS>>>,
         "Can only shift integers");

      using DOUT = Decay<TypeOf<OUT>>;
      using REGISTER = Inner::Register<LHS, RHS, OUT>;

      return Inner::Evaluate2<0, REGISTER, OUT>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return Inner::ShiftLeft<DOUT>(lhs, rhs);
         },
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            // Well defined condition in SIMD calls, that is otherwise  
            // undefined behavior by C++ standard                       
            return rhs < DOUT {sizeof(DOUT) * 8} and rhs >= 0
               ? lhs << rhs : 0;
         }
      );
   }

   /// Shift bits left, and force output to desired place                     
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   ///   @attention this differs from C++'s undefined behavior when shifting  
   ///      by less than zero, or by a number larger than the bitcount.       
   ///      The SIMD operations define this behavior very well, by just       
   ///      defaulting to zero. It is our responsibility to keep this         
   ///      behavior consistent across C++ and SIMD, so the fallback routine  
   ///      has additional overhead for checking the rhs range and zeroing.   
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT> LANGULUS(INLINED)
   constexpr void ShiftLeft(const LHS& lhs, const RHS& rhs, OUT& output) noexcept {
      IF_CONSTEXPR() {
         StoreConstexpr(ShiftLeftConstexpr<LHS, RHS, OUT>(lhs, rhs), output);
      }
      else Store(ShiftLeftDynamic<LHS, RHS, OUT>(lhs, rhs), output);
   }

   /// Shift bits left                                                        
   ///   @tparam LHS - left array, scalar, or register (deducible)            
   ///   @tparam RHS - right array, scalar, or register (deducible)           
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<CT::NotSemantic LHS, CT::NotSemantic RHS, CT::NotSemantic OUT = std::array<Lossless<Decay<TypeOf<LHS>>, Decay<TypeOf<RHS>>>, OverlapCounts<LHS, RHS>()>>
   LANGULUS(INLINED)
   constexpr OUT ShiftLeft(const LHS& lhs, const RHS& rhs) noexcept {
      OUT out;
      ShiftLeft(lhs, rhs, out);
      return out;
   }

} // namespace Langulus::SIMD