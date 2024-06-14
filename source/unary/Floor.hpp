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
      constexpr Unsupported FloorSIMD(CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Get floored values via SIMD                                         
      ///   @param value - the register                                       
      ///   @return the floored values                                        
      LANGULUS(INLINED)
      auto FloorSIMD(CT::SIMD auto value) noexcept {
         using R = decltype(value);
         using T = TypeOf<R>;
         static_assert(CT::Real<T>,
            "Suboptimal and pointless for whole numbers");
         (void)value;

         #if LANGULUS_COMPILER(CLANG) and LANGULUS(DEBUG)
            // WORKAROUND for a Clang bug, see:                         
            // https://github.com/simd-everywhere/simde/issues/1014     
            //TODO hopefully it is fixed in the future                  
            return Unsupported {};
         #else
            if constexpr (CT::SIMD128<R>) {
               if      constexpr (CT::Float<T>)    return R {simde_mm_floor_ps   (value)};
               else if constexpr (CT::Double<T>)   return R {simde_mm_floor_pd   (value)};
               else LANGULUS_ERROR("Unsupported type for 16-byte package");
            }
            else if constexpr (CT::SIMD256<R>) {
               if      constexpr (CT::Float<T>)    return R {simde_mm256_floor_ps(value)};
               else if constexpr (CT::Double<T>)   return R {simde_mm256_floor_pd(value)};
               else LANGULUS_ERROR("Unsupported type for 32-byte package");
            }
            else if constexpr (CT::SIMD512<R>) {
               if      constexpr (CT::Float<T>)    return R {simde_mm512_floor_ps(value)};
               else if constexpr (CT::Double<T>)   return R {simde_mm512_floor_pd(value)};
               else LANGULUS_ERROR("Unsupported type for 64-byte package");
            }
            else LANGULUS_ERROR("Unsupported type");
         #endif
      }
      
      /// Get floored values as constexpr, if possible                        
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the floored scalar/vector                                 
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto FloorConstexpr(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value, nullptr,
            []<class E>(const E& f) noexcept -> E {
               static_assert(CT::Real<E>, "Pointless for whole numbers");
               // std::floor isn't constexpr :(                         
               //TODO waiting for C++23 support                         
               const int64_t i = static_cast<int64_t>(f);
               return static_cast<E>(f < i ? i - 1 : i);
            }
         );
      }
   
      /// Get floored values as a register, if possible                       
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the floored scalar/vector/register                        
      template<CT::NotSemantic FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Floor(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value,
            []<class R>(const R& v) noexcept {
               LANGULUS_SIMD_VERBOSE("Flooring (SIMD) as ", NameOf<R>());
               return FloorSIMD(v);
            },
            []<class E>(const E& v) noexcept -> E {
               static_assert(CT::Real<E>, "Pointless for whole numbers");
               LANGULUS_SIMD_VERBOSE("Flooring (Fallback) ", v, " (", NameOf<E>(), ")");
               return std::floor(v);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_UNARY_API(Floor)

} // namespace Langulus::SIMD