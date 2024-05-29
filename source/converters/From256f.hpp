///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "From256f_To128f.hpp"
#include "From256f_To128d.hpp"
#include "From256f_To128i.hpp"
#include "From256f_To256d.hpp"
#include "From256f_To256i.hpp"

#if LANGULUS_SIMD(512BIT)
   #include "From256f_To512d.hpp"
   #include "From256f_To512i.hpp"
#endif

namespace Langulus::SIMD::Inner
{

   /// Convert __m256 to any other register                                   
   ///   @tparam TO - the desired element type                                
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::SIMD REGISTER> LANGULUS(INLINED)
   auto ConvertFrom256f(const simde__m256& v) noexcept {
      if constexpr (CT::SIMD128f<REGISTER>)
         return ConvertFrom256f_To128f(v);
      else if constexpr (CT::SIMD128d<REGISTER>)
         return ConvertFrom256f_To128d(v);
      else if constexpr (CT::SIMD128i<REGISTER>)
         return ConvertFrom256f_To128i<TO>(v);
      else if constexpr (CT::SIMD256i<REGISTER>)
         return ConvertFrom256f_To256i<TO>(v);
      else if constexpr (CT::SIMD256d<REGISTER>)
         return ConvertFrom256f_To256d(v);
      else
         LANGULUS_ERROR("Can't convert from __m256 to unsupported");
   }

} // namespace Langulus::SIMD