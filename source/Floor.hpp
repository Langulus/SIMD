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
      constexpr Unsupported FloorSIMD(const T&) noexcept {
         return {};
      }

      /// Get floored values via SIMD                                         
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param value - the register                                       
      ///   @return the floored values                                        
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto FloorSIMD(UNUSED() const REGISTER& value) noexcept {
         static_assert(CT::Real<T>, "Suboptimal and pointless for whole numbers");

      #if LANGULUS_COMPILER(CLANG) and LANGULUS(DEBUG)
         // WORKAROUND for a Clang bug, see:                            
         // https://github.com/simd-everywhere/simde/issues/1014        
         //TODO hopefully it is fixed in the future                     
         return Unsupported {};
      #else
      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm_floor_ps(value);
            else if constexpr (CT::Double<T>)
               return simde_mm_floor_pd(value);
            else
               LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else
      #endif
         
      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm256_floor_ps(value);
            else if constexpr (CT::Double<T>)
               return simde_mm256_floor_pd(value);
            else
               LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm512_floor_ps(value);
            else if constexpr (CT::Double<T>)
               return simde_mm512_floor_pd(value);
            else
               LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else
      #endif
            LANGULUS_ERROR("Unsupported type");
      #endif
      }
      
      /// Floor (constexpr, no SIMD)                                          
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return array/scalar                                              
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto FloorConstexpr(const auto& value) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;

         return Evaluate1<0, Unsupported, OUT>(
            value, nullptr,
            [](const DOUT& f) noexcept -> DOUT {
               static_assert(CT::Real<DOUT>, "Pointless for whole numbers");
               // std::floor isn't constexpr :(                         
               //TODO waiting for C++23 support                         
               const int64_t i = static_cast<int64_t>(f);
               return static_cast<DOUT>(f < i ? i - 1 : i);
            }
         );
      }
   
      /// Floor (SIMD)                                                        
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto Floor(const auto& value) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;
         using REGISTER = ToSIMD<decltype(value), OUT>;

         return Evaluate1<0, REGISTER, OUT>(
            value,
            [](const REGISTER& v) noexcept {
               LANGULUS_SIMD_VERBOSE("Flooring (SIMD) as ", NameOf<REGISTER>());
               return FloorSIMD<DOUT>(v);
            },
            [](const DOUT& v) noexcept -> DOUT {
               static_assert(CT::Real<DOUT>, "Pointless for whole numbers");
               LANGULUS_SIMD_VERBOSE("Flooring (Fallback) ", v, " (", NameOf<DOUT>(), ")");
               return std::floor(v);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_UNARY_API(Floor)

} // namespace Langulus::SIMD