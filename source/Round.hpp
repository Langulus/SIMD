///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "Fill.hpp"
#include "Convert.hpp"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Used to detect missing SIMD routine                                 
      template<CT::Decayed, CT::NotSIMD T> LANGULUS(INLINED)
      constexpr Unsupported RoundSIMD(const T&) noexcept {
         return {};
      }

      /// Get rounded values via SIMD                                         
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param value - the array                                          
      ///   @return the floored values                                        
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto RoundSIMD(UNUSED() const REGISTER& value) noexcept {
         static_assert(CT::Real<T>, "Suboptimal for unreal numbers");

         #if LANGULUS_SIMD(128BIT)
            constexpr auto STYLE = SIMDE_MM_FROUND_TO_NEAREST_INT | SIMDE_MM_FROUND_NO_EXC;

            if constexpr (CT::SIMD128<REGISTER>) {
               if constexpr (CT::Float<T>)
                  return simde_mm_round_ps(value, STYLE);
               else if constexpr (CT::Double<T>)
                  return simde_mm_round_pd(value, STYLE);
               else LANGULUS_ERROR("Unsupported type for 16-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(256BIT)
            if constexpr (CT::SIMD256<REGISTER>) {
               if constexpr (CT::Float<T>)
                  return simde_mm256_round_ps(value, STYLE);
               else if constexpr (CT::Double<T>)
                  return simde_mm256_round_pd(value, STYLE);
               else LANGULUS_ERROR("Unsupported type for 32-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(512BIT)
            if constexpr (CT::SIMD512<REGISTER>) {
               if constexpr (CT::Float<T>)
                  return simde_mm512_roundscale_ps(value, STYLE);
               else if constexpr (CT::Double<T>)
                  return simde_mm512_roundscale_pd(value, STYLE);
               else LANGULUS_ERROR("Unsupported type for 64-byte package");
            }
            else
         #endif

         LANGULUS_ERROR("Unsupported type");
      }

      /// Get rounded values as constexpr, if possible                        
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return array/scalar                                              
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto RoundConstexpr(const auto& value) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;

         return Evaluate1<0, Unsupported, OUT>(
            value, nullptr,
            [](const DOUT& f) noexcept -> DOUT {
               static_assert(CT::Signed<DOUT>, "Pointless for unsigned numbers");
               return std::round(f);
            }
         );
      }
   
      /// Get rounded values as a register, if possible                       
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto Round(const auto& value) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;
         using REGISTER = ToSIMD<decltype(value), OUT>;

         return Evaluate1<0, REGISTER, OUT>(
            value,
            [](const REGISTER& v) noexcept {
               LANGULUS_SIMD_VERBOSE("Rounded (SIMD) as ", NameOf<REGISTER>());
               return RoundSIMD<DOUT>(v);
            },
            [](const DOUT& v) noexcept -> DOUT {
               static_assert(CT::Signed<DOUT>, "Pointless for unsigned numbers");
               LANGULUS_SIMD_VERBOSE("Rounded (Fallback) ", v, " (", NameOf<DOUT>(), ")");
               return std::round(v);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_UNARY_API(Round)

} // namespace Langulus::SIMD