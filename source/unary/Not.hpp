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
      constexpr Unsupported NotSIMD(CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Bitwise NOT using registers                                         
      ///   @param lhs - register                                             
      ///   @return the resulting register                                    
      template<CT::SIMD R> LANGULUS(INLINED)
      R NotSIMD(R lhs) noexcept {
         return not lhs;
      }

      /// Get reversed values as constexpr, if possible                       
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector to operate on                        
      ///   @return the notted scalar/vector                                  
      template<CT::NoIntent FORCE_OUT = void> LANGULUS(INLINED)
      constexpr auto NotConstexpr(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value, nullptr,
            []<class E>(const E& f) noexcept -> E {
               return ~f;
            }
         );
      }
   
      /// Get not values as a register, if possible                           
      ///   @tparam FORCE_OUT - the desired element type (lossless if void)   
      ///   @patam value - scalar/vector/register to operate on               
      ///   @return the notted scalar/vector/register                         
      template<CT::NoIntent FORCE_OUT = void> LANGULUS(INLINED)
      auto Not(const auto& value) noexcept {
         return AttemptUnary<0, FORCE_OUT>(value,
            []<class R>(const R& v) noexcept {
               LANGULUS_SIMD_VERBOSE("Not (SIMD) as ", NameOf<R>());
               return NotSIMD(v);
            },
            []<class E>(const E& v) noexcept -> E {
               return ~v;
            }
         );
      }

   } // namespace Langulus::SIMD::Inner

   LANGULUS_SIMD_ARITHMETHIC_UNARY_API(Not)

} // namespace Langulus::SIMD