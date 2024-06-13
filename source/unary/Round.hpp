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
      LANGULUS(INLINED)
      constexpr Unsupported RoundSIMD(CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Get rounded values via SIMD                                         
      ///   @param value - the register                                       
      ///   @return the rounded values                                        
      LANGULUS(INLINED)
      auto RoundSIMD(CT::SIMD auto value) noexcept {
         using R = decltype(value);
         using T = TypeOf<R>;
         static_assert(CT::Real<T>,
            "Suboptimal and pointless for whole numbers");
         (void)value;

         constexpr auto STYLE = SIMDE_MM_FROUND_TO_NEAREST_INT | SIMDE_MM_FROUND_NO_EXC;

         if constexpr (CT::SIMD128<R>) {
            if      constexpr (CT::Float<T>)    return R {simde_mm_round_ps(value, STYLE)};
            else if constexpr (CT::Double<T>)   return R {simde_mm_round_pd(value, STYLE)};
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if      constexpr (CT::Float<T>)    return R {simde_mm256_round_ps(value, STYLE)};
            else if constexpr (CT::Double<T>)   return R {simde_mm256_round_pd(value, STYLE)};
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if      constexpr (CT::Float<T>)    return R {simde_mm512_roundscale_ps(value, STYLE)};
            else if constexpr (CT::Double<T>)   return R {simde_mm512_roundscale_pd(value, STYLE)};
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else LANGULUS_ERROR("Unsupported type");
      }
      
      /// Get rounded values as constexpr, if possible                        
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the rounded scalar/vector                                 
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto RoundConstexpr(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value, nullptr,
            []<class E>(const E& f) noexcept -> E {
               static_assert(CT::Signed<E>, "Pointless for unsigned numbers");
               return std::round(f);
            }
         );
      }
   
      /// Get rounded values as a register, if possible                       
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the rounded scalar/vector/register                        
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Round(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value,
            []<class R>(const R& v) noexcept {
               LANGULUS_SIMD_VERBOSE("Rounded (SIMD) as ", NameOf<R>());
               return RoundSIMD(v);
            },
            []<class E>(const E& v) noexcept -> E {
               static_assert(CT::Signed<E>, "Pointless for unsigned numbers");
               LANGULUS_SIMD_VERBOSE("Rounded (Fallback) ", v, " (", NameOf<E>(), ")");
               return std::round(v);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_UNARY_API(Round)

} // namespace Langulus::SIMD