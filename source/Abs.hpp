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
      constexpr Unsupported AbsSIMD(const T&) noexcept {
         return {};
      }

      /// Get absolute values via SIMD                                        
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - the register to operate on (deducible)         
      ///   @param v - the array                                              
      ///   @return the absolute values                                       
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto AbsSIMD(UNUSED() const REGISTER& v) noexcept {
         static_assert(CT::Signed<T>, "Suboptimal and pointless for unsigned values");

         #if LANGULUS_SIMD(128BIT)
            if constexpr (CT::SIMD128<REGISTER>) {
               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm_abs_epi8(v);
               else if constexpr (CT::SignedInteger16<T>)
                  return simde_mm_abs_epi16(v);
               else if constexpr (CT::SignedInteger32<T>)
                  return simde_mm_abs_epi32(v);
               else if constexpr (CT::SignedInteger64<T>)
                  return simde_mm_abs_epi64(v);
               else if constexpr (CT::Float<T>)
                  return simde_mm_andnot_ps(simde_mm_set1_ps(-0.0F), v);
               else if constexpr (CT::Double<T>)
                  return simde_mm_andnot_pd(simde_mm_set1_pd(-0.0F), v);
               else LANGULUS_ERROR("Unsupported type for 16-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(256BIT)
            if constexpr (CT::SIMD256<REGISTER>) {
               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm256_abs_epi8(v);
               else if constexpr (CT::SignedInteger16<T>)
                  return simde_mm256_abs_epi16(v);
               else if constexpr (CT::SignedInteger32<T>)
                  return simde_mm256_abs_epi32(v);
               else if constexpr (CT::SignedInteger64<T>)
                  return simde_mm256_abs_epi64(v);
               else if constexpr (CT::Float<T>)
                  return simde_mm256_andnot_ps(simde_mm256_set1_ps(-0.0F), v);
               else if constexpr (CT::Double<T>)
                  return simde_mm256_andnot_pd(simde_mm256_set1_pd(-0.0F), v);
               else LANGULUS_ERROR("Unsupported type for 32-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(512BIT)
            if constexpr (CT::SIMD512<REGISTER>) {
               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm512_abs_epi8(v);
               else if constexpr (CT::SignedInteger16<T>)
                  return simde_mm512_abs_epi16(v);
               else if constexpr (CT::SignedInteger32<T>)
                  return simde_mm512_abs_epi32(v);
               else if constexpr (CT::SignedInteger64<T>)
                  return simde_mm512_abs_epi64(v);
               else if constexpr (CT::Float<T>)
                  return simde_mm512_andnot_ps(simde_mm512_set1_ps(-0.0F), v);
               else if constexpr (CT::Double<T>)
                  return simde_mm512_andnot_pd(simde_mm512_set1_pd(-0.0F), v);
               else LANGULUS_ERROR("Unsupported type for 64-byte package");
            }
            else
         #endif

         LANGULUS_ERROR("Unsupported type");
      }
      
      /// Get absolute values as constexpr, if possible                       
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return array/scalar                                              
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto AbsConstexpr(const auto& value) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;

         return Evaluate1<0, Unsupported, OUT>(
            value, nullptr,
            [](const DOUT& f) noexcept -> DOUT {
               static_assert(CT::Signed<DOUT>, "Pointless for unsigned numbers");
               return f < DOUT {0} ? -f : f;
            }
         );
      }
   
      /// Get absolute values as a register, if possible                      
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto Abs(const auto& value) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;
         using REGISTER = ToSIMD<decltype(value), OUT>;

         return Evaluate1<0, REGISTER, OUT>(
            value,
            [](const REGISTER& v) noexcept {
               LANGULUS_SIMD_VERBOSE("Absolute (SIMD) as ", NameOf<REGISTER>());
               return AbsSIMD<DOUT>(v);
            },
            [](const DOUT& v) noexcept -> DOUT {
               static_assert(CT::Signed<DOUT>, "Pointless for unsigned numbers");
               LANGULUS_SIMD_VERBOSE("Absolute (Fallback) |", v, "| (", NameOf<DOUT>(), ")");
               return std::abs(v);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_UNARY_API(Abs)

} // namespace Langulus::SIMD