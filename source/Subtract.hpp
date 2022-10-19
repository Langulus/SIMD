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
   LANGULUS(ALWAYSINLINE) constexpr auto SubtractInner(CT::NotSupported auto, CT::NotSupported auto) noexcept {
      return CT::Inner::NotSupported{};
   }

   /// Subtract two arrays using SIMD                                         
   ///   @tparam T - the type of the array element                            
   ///   @tparam S - the size of the array                                    
   ///   @tparam REGISTER - type of register we're operating with             
   ///   @param lhs - the left-hand-side array                                
   ///   @param rhs - the right-hand-side array                               
   ///   @return the subtracted elements as a register                        
   template<class T, Count S, CT::TSIMD REGISTER>
   LANGULUS(ALWAYSINLINE) auto SubtractInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
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
            else if constexpr (CT::RealSP<T>)
               return simde_mm_sub_ps(lhs, rhs);
            else if constexpr (CT::RealDP<T>)
               return simde_mm_sub_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::SubtractInner of 16-byte package");
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
            else if constexpr (CT::RealSP<T>)
               return simde_mm256_sub_ps(lhs, rhs);
            else if constexpr (CT::RealDP<T>)
               return simde_mm256_sub_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::SubtractInner of 32-byte package");
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
            else if constexpr (CT::RealSP<T>)
               return simde_mm512_sub_ps(lhs, rhs);
            else if constexpr (CT::RealDP<T>)
               return simde_mm512_sub_pd(lhs, rhs);
            else LANGULUS_ERROR("Unsupported type for SIMD::SubtractInner of 64-byte package");
         }
         else
      #endif

      LANGULUS_ERROR("Unsupported type for SIMD::SubtractInner");
   }

   ///                                                                        
   template<class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) auto Subtract(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using REGISTER = CT::Register<LHS, RHS>;
      using LOSSLESS = Lossless<LHS, RHS>;
      constexpr auto S = OverlapCount<LHS, RHS>();
      return AttemptSIMD<0, REGISTER, LOSSLESS>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return SubtractInner<LOSSLESS, S>(lhs, rhs);
         },
         [](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept -> LOSSLESS {
            return lhs - rhs;
         }
      );
   }

   ///                                                                        
   template<class LHS, class RHS, class OUT>
   LANGULUS(ALWAYSINLINE) void Subtract(const LHS& lhs, const RHS& rhs, OUT& output) noexcept {
      GeneralStore(Subtract<LHS, RHS>(lhs, rhs), output);
   }

   ///                                                                        
   template<CT::Vector WRAPPER, class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) WRAPPER SubtractWrap(const LHS& lhs, const RHS& rhs) noexcept {
      WRAPPER result;
      Subtract<LHS, RHS>(lhs, rhs, result.mArray);
      return result;
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
