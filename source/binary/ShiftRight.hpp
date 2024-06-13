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
      constexpr Unsupported ShiftRightSIMD(CT::NotSIMD auto, CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Shift right using registers                                         
      ///   @attention this differs from C++'s undefined behavior when        
      ///      shifting by less than zero, or by a number larger than the     
      ///      bitcount. The SIMD operations define this behavior very well,  
      ///      by just defaulting to zero. It is our responsibility to keep   
      ///      this behavior consistent across C++ and SIMD, so the fallback  
      ///      routine has additional overhead for checking the rhs range and 
      ///      zeroing.                                                       
      ///   @param lhs - left register                                        
      ///   @param rhs - right register                                       
      ///   @return the resulting register                                    
      template<CT::SIMD R> NOD() LANGULUS(INLINED)
      auto ShiftRightSIMD(R lhs, R rhs) noexcept {
         using T = TypeOf<R>;
         static_assert(CT::IntegerX<T>, "Can only shift integers");
         (void)lhs; (void)rhs;

         if constexpr (CT::SIMD128<R>) {
            if constexpr (CT::Integer8<T>) {
               #if LANGULUS_SIMD(512BIT)
                  // Optimal                                            
                  return R {lgls_pack_epi16(
                     simde_mm_srlv_epi16(lhs.UnpackLo(), rhs.UnpackLo()),
                     simde_mm_srlv_epi16(lhs.UnpackHi(), rhs.UnpackHi())
                  )};
               #elif LANGULUS_SIMD(256BIT)
                  // Not optimal, must be unpacked once more for AVX2   
                  auto lhs32_1 = lhs.UnpackLo().UnpackLo();
                  auto lhs32_2 = lhs.UnpackHi().UnpackHi();
                  auto rhs32_1 = rhs.UnpackLo().UnpackLo();
                  auto rhs32_2 = rhs.UnpackHi().UnpackHi();

                  lhs32_1 = simde_mm_srlv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm_srlv_epi32(lhs32_2, rhs32_2);
                  auto lo = lgls_pack_epi32(lhs32_1, lhs32_2);

                  lhs32_1 = simde_mm_srlv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm_srlv_epi32(lhs32_2, rhs32_2);
                  auto hi = lgls_pack_epi32(lhs32_1, lhs32_2);

                  return R {lgls_pack_epi16(lo, hi)};
               #else
                  return Unsupported{}; //TODO
               #endif
            }
            else if constexpr (CT::Integer16<T>) {
               #if LANGULUS_SIMD(512BIT)
                  // Optimal                                            
                  return R {simde_mm_srlv_epi16(lhs, rhs)};
               #elif LANGULUS_SIMD(256BIT)
                  // Not optimal, must be unpacked for AVX2             
                  auto lhs32_1 = lhs.UnpackLo();
                  auto lhs32_2 = lhs.UnpackHi();
                  auto rhs32_1 = rhs.UnpackLo();
                  auto rhs32_2 = rhs.UnpackHi();

                  lhs32_1 = simde_mm_srlv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm_srlv_epi32(lhs32_2, rhs32_2);
                  return R {lgls_pack_epi32(lhs32_1, lhs32_2)};
               #else
                  return Unsupported{}; //TODO
               #endif
            }
            else if constexpr (CT::Integer32<T>) {
               #if LANGULUS_SIMD(256BIT)
                  return R {simde_mm_srlv_epi32(lhs, rhs)};
               #else
                  return Unsupported {}; //TODO
               #endif
            }
            else if constexpr (CT::Integer64<T>) {
               #if LANGULUS_SIMD(256BIT)
                  return R {simde_mm_srlv_epi64(lhs, rhs)};
               #else
                  return Unsupported {}; //TODO
               #endif
            }
            else LANGULUS_ERROR("Unsupported type for SIMD::ShiftRightInner of 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if constexpr (CT::Integer8<T>) {
               auto lhs1 = lhs.UnpackLo();
               auto lhs2 = lhs.UnpackHi();
               auto rhs1 = rhs.UnpackLo();
               auto rhs2 = rhs.UnpackHi();

               #if LANGULUS_SIMD(512BIT)
                  // Optimal                                            
                  lhs1 = simde_mm256_srlv_epi16(lhs1, rhs1);
                  lhs2 = simde_mm256_srlv_epi16(lhs2, rhs2);
                  return R {lgls_pack_epi16(lhs1, lhs2)};
               #else
                  // Not optimal, must be unpacked once more for AVX2   
                  auto lhs32_1 = rhs1.UnpackLo();
                  auto lhs32_2 = rhs1.UnpackLo();
                  auto rhs32_1 = rhs1.UnpackLo();
                  auto rhs32_2 = rhs1.UnpackLo();

                  lhs32_1 = simde_mm256_srlv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm256_srlv_epi32(lhs32_2, rhs32_2);
                  lhs1 = lgls_pack_epi32(lhs32_1, lhs32_2);

                  lhs32_1 = rhs2.UnpackLo();
                  lhs32_2 = rhs2.UnpackLo();
                  rhs32_1 = rhs2.UnpackLo();
                  rhs32_2 = rhs2.UnpackLo();

                  lhs32_1 = simde_mm256_srlv_epi32(lhs32_1, rhs32_1);
                  lhs32_2 = simde_mm256_srlv_epi32(lhs32_2, rhs32_2);
                  lhs2 = lgls_pack_epi32(lhs32_1, lhs32_2);

                  return R {lgls_pack_epi16(lhs1, lhs2)};
               #endif
            }
            else if constexpr (CT::Integer16<T>) {
               #if LANGULUS_SIMD(512BIT)
                  // Optimal                                            
                  return simde_mm256_srlv_epi16(lhs, rhs);
               #else
                  // Not optimal, must be unpacked for AVX2             
                  auto lhs1 = lhs.UnpackLo();
                  auto lhs2 = lhs.UnpackHi();
                  auto rhs1 = rhs.UnpackLo();
                  auto rhs2 = rhs.UnpackHi();

                  lhs1 = simde_mm256_srlv_epi32(lhs1, rhs1);
                  lhs2 = simde_mm256_srlv_epi32(lhs2, rhs2);
                  return R {lgls_pack_epi32(lhs1, lhs2)};
               #endif
            }
            else if constexpr (CT::Integer32<T>)         return R {simde_mm256_srlv_epi32(lhs, rhs)};
            else if constexpr (CT::Integer64<T>)         return R {simde_mm256_srlv_epi64(lhs, rhs)};
            else LANGULUS_ERROR("Unsupported type for SIMD::ShiftRightInner of 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if constexpr (CT::Integer8<T>) {
               auto lhs1 = lhs.UnpackLo();
               auto lhs2 = lhs.UnpackHi();
               auto rhs1 = rhs.UnpackLo();
               auto rhs2 = rhs.UnpackHi();

               lhs1 = simde_mm512_srlv_epi16(lhs1, rhs1);
               lhs2 = simde_mm512_srlv_epi16(lhs2, rhs2);
               return R {lgls_pack_epi16(lhs1, lhs2)};
            }
            else if constexpr (CT::Integer16<T>)         return R {simde_mm512_srlv_epi16(lhs, rhs)};
            else if constexpr (CT::Integer32<T>)         return R {simde_mm512_srlv_epi32(lhs, rhs)};
            else if constexpr (CT::Integer64<T>)         return R {simde_mm512_srlv_epi64(lhs, rhs)};
            else LANGULUS_ERROR("Unsupported type for SIMD::ShiftRightInner of 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type for SIMD::ShiftRightInner");
      }
      
      /// Bitwise right shift values as constexpr, if possible                
      ///   @attention this differs from C++'s undefined behavior when        
      ///      shifting by less than zero, or by a number larger than the     
      ///      bitcount. The SIMD operations define this behavior very well,  
      ///      by just defaulting to zero. It is our responsibility to keep   
      ///      this behavior consistent across C++ and SIMD, so the fallback  
      ///      routine has additional overhead for checking the rhs range and 
      ///      zeroing.                                                       
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the shifted scalar/vector                                 
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto ShiftRightConstexpr(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs, nullptr,
            []<class E>(const E& l, const E& r) noexcept -> E {
               // Well defined condition in SIMD calls, that is         
               // otherwise undefined behavior by C++ standard          
               static_assert(CT::IntegerX<E>, "Can only shift integers");
               return r < E {sizeof(E) * 8} and r >= 0
                  ? l >> r : 0;
            }
         );
      }
   
      /// Bitwise right shift values as a register, if possible               
      ///   @attention this differs from C++'s undefined behavior when        
      ///      shifting by less than zero, or by a number larger than the     
      ///      bitcount. The SIMD operations define this behavior very well,  
      ///      by just defaulting to zero. It is our responsibility to keep   
      ///      this behavior consistent across C++ and SIMD, so the fallback  
      ///      routine has additional overhead for checking the rhs range and 
      ///      zeroing.                                                       
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the shifted scalar/vector/register                        
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto ShiftRight(const auto& lhs, const auto& rhs) noexcept {
         return AttemptBinary<0, FORCE_OUT>(lhs, rhs,
            []<class R>(const R& l, const R& r) noexcept {
               return ShiftRightSIMD(l, r);
            },
            []<class E>(const E& l, const E& r) noexcept -> E {
               // Well defined condition in SIMD calls, that is         
               // otherwise undefined behavior by C++ standard          
               static_assert(CT::IntegerX<E>, "Can only shift integers");
               return r < E {sizeof(E) * 8} and r >= 0
                  ? l >> r : 0;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_API(ShiftRight)

} // namespace Langulus::SIMD