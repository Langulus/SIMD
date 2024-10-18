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
      constexpr Unsupported CeilSIMD(CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Get ceiled values via SIMD                                          
      ///   @param value - the register                                       
      ///   @return the ceiled values                                         
      LANGULUS(INLINED)
      auto CeilSIMD(CT::SIMD auto value) noexcept {
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
               if      constexpr (CT::Float<T>)    return R {simde_mm_ceil_ps   (value)};
               else if constexpr (CT::Double<T>)   return R {simde_mm_ceil_pd   (value)};
               else static_assert(false, "Unsupported type for 16-byte package");
            }
            else if constexpr (CT::SIMD256<R>) {
               if      constexpr (CT::Float<T>)    return R {simde_mm256_ceil_ps(value)};
               else if constexpr (CT::Double<T>)   return R {simde_mm256_ceil_pd(value)};
               else static_assert(false, "Unsupported type for 32-byte package");
            }
            else if constexpr (CT::SIMD512<R>) {
               if      constexpr (CT::Float<T>)    return R {simde_mm512_ceil_ps(value)};
               else if constexpr (CT::Double<T>)   return R {simde_mm512_ceil_pd(value)};
               else static_assert(false, "Unsupported type for 64-byte package");
            }
            else static_assert(false, "Unsupported type");
         #endif
      }
      
      /// Get ceiled values as constexpr, if possible                         
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the ceiled scalar/vector                                  
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      constexpr auto CeilConstexpr(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value, nullptr,
            []<class E>(const E& f) noexcept -> E {
               static_assert(CT::Real<E>, "Pointless for whole numbers");
               // std::ceil isn't constexpr :(                          
               //TODO waiting for C++23 support                         
               const int64_t i = static_cast<int64_t>(f);
               return static_cast<E>(f > i ? i + 1 : i);
            }
         );
      }
   
      /// Get ceiled values as a register, if possible                        
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the ceiled scalar/vector/register                         
      template<CT::NoIntent FORCE_OUT = void> NOD() LANGULUS(INLINED)
      auto Ceil(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value,
            []<class R>(const R& v) noexcept {
               LANGULUS_SIMD_VERBOSE("Ceiling (SIMD) as ", NameOf<R>());
               return CeilSIMD(v);
            },
            []<class E>(const E& v) noexcept -> E {
               static_assert(CT::Real<E>, "Pointless for whole numbers");
               LANGULUS_SIMD_VERBOSE("Ceiling (Fallback) ", v, " (", NameOf<E>(), ")");
               return std::ceil(v);
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_UNARY_API(Ceil)

} // namespace Langulus::SIMD