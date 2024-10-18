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
      constexpr Unsupported AbsSIMD(CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Get absolute values via SIMD                                        
      ///   @param v - the register                                           
      ///   @return the absolute values                                       
      NOD() LANGULUS(INLINED)
      auto AbsSIMD(CT::SIMD auto v) noexcept {
         using R = decltype(v);
         using T = TypeOf<R>;
         static_assert(CT::Signed<T>,
            "Suboptimal and pointless for unsigned values");
         (void)v;

         if constexpr (CT::SIMD128<R>) {
            if      constexpr (CT::SignedInteger8<T>)  return R {simde_mm_abs_epi8(v)};
            else if constexpr (CT::SignedInteger16<T>) return R {simde_mm_abs_epi16(v)};
            else if constexpr (CT::SignedInteger32<T>) return R {simde_mm_abs_epi32(v)};
            else if constexpr (CT::SignedInteger64<T>) return R {simde_mm_abs_epi64(v)};
            else if constexpr (CT::Float<T>)           return R {simde_mm_andnot_ps(simde_mm_set1_ps(-0.0F), v)};
            else if constexpr (CT::Double<T>)          return R {simde_mm_andnot_pd(simde_mm_set1_pd(-0.0F), v)};
            else static_assert(false, "Unsupported type for 16-byte package");
         }
         else if constexpr (CT::SIMD256<R>) {
            if      constexpr (CT::SignedInteger8<T>)  return R {simde_mm256_abs_epi8(v)};
            else if constexpr (CT::SignedInteger16<T>) return R {simde_mm256_abs_epi16(v)};
            else if constexpr (CT::SignedInteger32<T>) return R {simde_mm256_abs_epi32(v)};
            else if constexpr (CT::SignedInteger64<T>) return R {simde_mm256_abs_epi64(v)};
            else if constexpr (CT::Float<T>)           return R {simde_mm256_andnot_ps(simde_mm256_set1_ps(-0.0F), v)};
            else if constexpr (CT::Double<T>)          return R {simde_mm256_andnot_pd(simde_mm256_set1_pd(-0.0F), v)};
            else static_assert(false, "Unsupported type for 32-byte package");
         }
         else if constexpr (CT::SIMD512<R>) {
            if      constexpr (CT::SignedInteger8<T>)  return R {simde_mm512_abs_epi8(v)};
            else if constexpr (CT::SignedInteger16<T>) return R {simde_mm512_abs_epi16(v)};
            else if constexpr (CT::SignedInteger32<T>) return R {simde_mm512_abs_epi32(v)};
            else if constexpr (CT::SignedInteger64<T>) return R {simde_mm512_abs_epi64(v)};
            else if constexpr (CT::Float<T>)           return R {simde_mm512_andnot_ps(simde_mm512_set1_ps(-0.0F), v)};
            else if constexpr (CT::Double<T>)          return R {simde_mm512_andnot_pd(simde_mm512_set1_pd(-0.0F), v)};
            else static_assert(false, "Unsupported type for 64-byte package");
         }
         else static_assert(false, "Unsupported type");
      }
      
      /// Get absolute values as constexpr, if possible                       
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the absolute scalar/vector                                
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto AbsConstexpr(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value, nullptr,
            []<class E>(const E& f) noexcept -> E {
               static_assert(CT::Signed<E>, "Pointless for unsigned numbers");
               return f < E {0} ? -f : f;
            }
         );
      }
   
      /// Get absolute values as a register, if possible                      
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the absolute scalar/vector/register                       
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Abs(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value,
            []<class R>(const R& v) noexcept {
               LANGULUS_SIMD_VERBOSE("Absolute (SIMD) as ", NameOf<R>());
               return AbsSIMD(v);
            },
            []<class E>(const E& v) noexcept -> E {
               static_assert(CT::Signed<E>, "Pointless for unsigned numbers");
               LANGULUS_SIMD_VERBOSE("Absolute (Fallback) |", v, "| (", NameOf<E>(), ")");
               return std::abs(v);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_UNARY_API(Abs)

} // namespace Langulus::SIMD