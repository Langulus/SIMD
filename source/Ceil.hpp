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
      constexpr Unsupported CeilSIMD(const T&) noexcept {
         return {};
      }

      /// Get ceiled values via SIMD                                          
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param value - the register                                       
      ///   @return the ceiled values                                         
      template<CT::Decayed T, CT::SIMD REGISTER> LANGULUS(INLINED)
      auto CeilSIMD(UNUSED() const REGISTER& value) noexcept {
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
               return simde_mm_ceil_ps(value);
            else if constexpr (CT::Double<T>)
               return simde_mm_ceil_pd(value);
            else
               LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else
      #endif
         
      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm256_ceil_ps(value);
            else if constexpr (CT::Double<T>)
               return simde_mm256_ceil_pd(value);
            else
               LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::Float<T>)
               return simde_mm512_ceil_ps(value);
            else if constexpr (CT::Double<T>)
               return simde_mm512_ceil_pd(value);
            else
               LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else
      #endif
            LANGULUS_ERROR("Unsupported type");
      #endif
      }
      
      /// Ceil (constexpr, no SIMD)                                           
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return array/scalar                                              
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      constexpr auto CeilConstexpr(const auto& value) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;

         return Evaluate1<0, Unsupported, OUT>(
            value, nullptr,
            [](const DOUT& f) noexcept -> DOUT {
               static_assert(CT::Real<DOUT>, "Pointless for whole numbers");
               // std::ceil isn't constexpr :(                          
               //TODO waiting for C++23 support                         
               const int64_t i = static_cast<int64_t>(f);
               return static_cast<DOUT>(f > i ? i + 1 : i);
            }
         );
      }
   
      /// Ceil (SIMD)                                                         
      ///   @tparam OUT - the desired element type (lossless by default)      
      ///   @return a register, if viable SIMD routine exists                 
      ///           or array/scalar if no viable SIMD routine exists          
      template<CT::NotSemantic OUT> NOD() LANGULUS(INLINED)
      auto CeilDynamic(const auto& value) noexcept {
         using DOUT = Decay<TypeOf<OUT>>;
         using REGISTER = ToSIMD<decltype(value), OUT>;

         return Evaluate1<0, REGISTER, OUT>(
            value,
            [](const REGISTER& v) noexcept {
               LANGULUS_SIMD_VERBOSE("Ceiling (SIMD) as ", NameOf<REGISTER>());
               return CeilSIMD<DOUT>(v);
            },
            [](const DOUT& v) noexcept -> DOUT {
               static_assert(CT::Real<DOUT>, "Pointless for whole numbers");
               LANGULUS_SIMD_VERBOSE("Ceiling (Fallback) ", v, " (", NameOf<DOUT>(), ")");
               return std::ceil(v);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   
   /// Ceil numbers, and force output to desired place                        
   ///   @tparam VAL - array, scalar, or register (deducible)                 
   ///   @tparam OUT - the desired element type (deducible)                   
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<class VAL, CT::NotSemantic OUT> LANGULUS(INLINED)
   constexpr void Ceil(const VAL& val, OUT& out) noexcept {
      IF_CONSTEXPR() {
      StoreConstexpr(
         Inner::CeilConstexpr<OUT>(DesemCast(val)), out);
      }
      else Store(
         Inner::CeilDynamic<OUT>(DesemCast(val)), out);
   }
      
   /// Ceil numbers                                                           
   ///   @tparam VAL - array, scalar, or register (deducible)                 
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<class VAL, CT::NotSemantic OUT = LosslessArray<VAL, VAL>>
   LANGULUS(INLINED)
   constexpr auto Ceil(const VAL& val) noexcept {
      OUT out;
      Ceil(DesemCast(val), out);
      return out;
   }

} // namespace Langulus::SIMD