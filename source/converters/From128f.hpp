///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "From128f_To128d.hpp"
#include "From128f_To128i.hpp"

#if LANGULUS_SIMD(256BIT)
   #include "From128f_To256d.hpp"
   #include "From128f_To256i.hpp"
#endif

namespace Langulus::SIMD::Inner
{

   /// Convert __m128 to any other register                                   
   ///   @tparam TO - the desired element type                                
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::SIMD REGISTER> LANGULUS(INLINED)
   auto ConvertFrom128f(const simde__m128& v) noexcept {
      if constexpr (CT::SIMD128d<REGISTER>)
         return ConvertFrom128f_To128d(v);
      else if constexpr (CT::SIMD128i<REGISTER>)
         return ConvertFrom128f_To128i<TO>(v);
      else
   #if LANGULUS_SIMD(256BIT)
      if constexpr (CT::SIMD256d<REGISTER>)
         return ConvertFrom128f_To256d(v);
      else if constexpr (CT::SIMD256i<REGISTER>)
         return ConvertFrom128f_To256i<TO>(v);
      else
   #endif
         LANGULUS_ERROR("Can't convert from __m128 to unsupported");
   }

} // namespace Langulus::SIMD