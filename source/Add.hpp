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
#include "Store.hpp"
#include "IgnoreWarningsPush.inl"

namespace Langulus::SIMD
{
      
   template<class, Count>
   LANGULUS(ALWAYSINLINE) constexpr auto AddInner(CT::NotSupported auto, CT::NotSupported auto) noexcept {
      return CT::Inner::NotSupported{};
   }

   /// Add two arrays using SIMD                                              
   ///   @tparam T - the type of the array element                            
   ///   @tparam S - the size of the array                                    
   ///   @tparam REGISTER - the register type (deducible)                     
   ///   @param lhs - the left-hand-side array                                
   ///   @param rhs - the right-hand-side array                               
   ///   @return the added elements as a register                             
   template<class T, Count S, CT::TSIMD REGISTER>
   LANGULUS(ALWAYSINLINE) auto AddInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>)
               return simde_mm_add_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)
               return simde_mm_adds_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)
               return simde_mm_add_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>)
               return simde_mm_adds_epu16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)
               return simde_mm_add_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)
               return simde_mm_add_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)
               return simde_mm_add_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm_add_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::AddInner of 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>)
               return simde_mm256_add_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)
               return simde_mm256_adds_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)
               return simde_mm256_add_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>)
               return simde_mm256_adds_epu16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)
               return simde_mm256_add_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)
               return simde_mm256_add_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)
               return simde_mm256_add_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm256_add_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::AddInner of 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::SignedInteger8<T>)
               return simde_mm512_add_epi8(lhs, rhs);
            else if constexpr (CT::UnsignedInteger8<T>)
               return simde_mm512_adds_epu8(lhs, rhs);
            else if constexpr (CT::SignedInteger16<T>)
               return simde_mm512_add_epi16(lhs, rhs);
            else if constexpr (CT::UnsignedInteger16<T>)
               return simde_mm512_adds_epu16(lhs, rhs);
            else if constexpr (CT::Integer32<T>)
               return simde_mm512_add_epi32(lhs, rhs);
            else if constexpr (CT::Integer64<T>)
               return simde_mm512_add_epi64(lhs, rhs);
            else if constexpr (CT::Float<T>)
               return simde_mm512_add_ps(lhs, rhs);
            else if constexpr (CT::Double<T>)
               return simde_mm512_add_pd(lhs, rhs);
            else
               LANGULUS_ERROR("Unsupported type for SIMD::AddInner of 64-byte package");
         }
         else
      #endif
         LANGULUS_ERROR("Unsupported type for SIMD::AddInner");
   }

   ///                                                                        
   template<class LHS, class RHS, class OUT = Lossless<LHS, RHS>>
   NOD() LANGULUS(ALWAYSINLINE) auto Add(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
      using DOUT = Decay<OUT>;
      using REGISTER = CT::Register<LHS, RHS, DOUT>;
      constexpr auto S = OverlapCount<LHS, RHS>();

      return AttemptSIMD<0, REGISTER, DOUT>(
         lhsOrig, rhsOrig, 
         [](const REGISTER& lhs, const REGISTER& rhs) noexcept {
            return AddInner<DOUT, S>(lhs, rhs);
         },
         [](const DOUT& lhs, const DOUT& rhs) noexcept -> DOUT {
            return lhs + rhs;
         }
      );
   }

   ///                                                                        
   template<class LHS, class RHS, class OUT>
   LANGULUS(ALWAYSINLINE) void Add(const LHS& lhs, const RHS& rhs, OUT& output) noexcept {
      GeneralStore(Add<LHS, RHS, OUT>(lhs, rhs), output);
   }

   ///                                                                        
   template<CT::Vector WRAPPER, class LHS, class RHS>
   NOD() LANGULUS(ALWAYSINLINE) WRAPPER AddWrap(const LHS& lhs, const RHS& rhs) noexcept {
      WRAPPER result;
      Add<LHS, RHS>(lhs, rhs, result.mArray);
      return result;
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
