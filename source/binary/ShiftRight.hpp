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
               #if LANGULUS_SIMD(256BIT) or LANGULUS_SIMD(512BIT)
                  auto lo = ShiftRightSIMD(lhs.UnpackLo(), rhs.UnpackLo());
                  auto hi = ShiftRightSIMD(lhs.UnpackHi(), rhs.UnpackHi());
                  return R {lgls_pack_epi16(lo, hi)};
               #else
                  return Unsupported {}; //TODO
               #endif
            }
            else if constexpr (CT::Integer16<T>) {
               #if LANGULUS_SIMD(512BIT)
                  return simde_mm_srlv_epi16(lhs, rhs);
               #elif LANGULUS_SIMD(256BIT)
                  auto lo = ShiftRightSIMD(lhs.UnpackLo(), rhs.UnpackLo());
                  auto hi = ShiftRightSIMD(lhs.UnpackHi(), rhs.UnpackHi());
                  return R {lgls_pack_epi32(lo, hi)};
               #else
                  return Unsupported {}; //TODO
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
            else static_assert(false, "Unsupported type for SIMD::ShiftRightInner of 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if constexpr (CT::Integer8<T>) {
               auto lo = ShiftRightSIMD(lhs.UnpackLo(), rhs.UnpackLo());
               auto hi = ShiftRightSIMD(lhs.UnpackHi(), rhs.UnpackHi());
               return R {lgls_pack_epi16(lo, hi)};
            }
            else if constexpr (CT::Integer16<T>) {
               #if LANGULUS_SIMD(512BIT)
                  return simde_mm256_sllv_epi16(lhs, rhs);
               #else
                  auto lo = ShiftRightSIMD(lhs.UnpackLo(), rhs.UnpackLo());
                  auto hi = ShiftRightSIMD(lhs.UnpackHi(), rhs.UnpackHi());
                  return R {lgls_pack_epi32(lo, hi)};
               #endif
            }
            else if constexpr (CT::Integer32<T>)         return R {simde_mm256_srlv_epi32(lhs, rhs)};
            else if constexpr (CT::Integer64<T>)         return R {simde_mm256_srlv_epi64(lhs, rhs)};
            else static_assert(false, "Unsupported type for SIMD::ShiftRightInner of 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if constexpr (CT::Integer8<T>) {
               auto lo = ShiftRightSIMD(lhs.UnpackLo(), rhs.UnpackLo());
               auto hi = ShiftRightSIMD(lhs.UnpackHi(), rhs.UnpackHi());
               return R {lgls_pack_epi16(lo, hi)};
            }
            else if constexpr (CT::Integer16<T>)         return R {simde_mm512_srlv_epi16(lhs, rhs)};
            else if constexpr (CT::Integer32<T>)         return R {simde_mm512_srlv_epi32(lhs, rhs)};
            else if constexpr (CT::Integer64<T>)         return R {simde_mm512_srlv_epi64(lhs, rhs)};
            else static_assert(false, "Unsupported type for SIMD::ShiftRightInner of 64-byte package");
         }
         else static_assert(false, "Unsupported type for SIMD::ShiftRightInner");
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
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
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
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
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