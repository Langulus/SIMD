///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "From256i_To128f.hpp"
#include "From256i_To128d.hpp"
#include "From256i_To128i.hpp"
#include "From256i_To256f.hpp"
#include "From256i_To256d.hpp"
#include "From256i_To256i.hpp"

#if LANGULUS_SIMD(512BIT)
   #include "From256i_To512f.hpp"
   #include "From256i_To512d.hpp"
   #include "From256i_To512i.hpp"
#endif


namespace Langulus::SIMD::Inner
{

   /// Convert __m256i to any other register                                  
   ///   @tparam TO - the desired element type                                
   ///   @tparam FROM - the previous element type, contained in REGISTER      
   ///                  (a 256i register can contain various kinds of ints)   
   ///   @tparam REGISTER - register to convert to                            
   ///   @param v - the input register                                        
   ///   @return the resulting register                                       
   template<CT::Decayed TO, CT::Decayed FROM, CT::SIMD REGISTER>
   LANGULUS(INLINED)
   auto ConvertFrom256i(const simde__m256i& v) noexcept {
      //                                                                
      // Converting FROM i8[32], u8[32], i16[16], u16[16]               
      //                 i32[8], u32[8], i64[4],  u64[4]                
      //                                                                
      if constexpr (CT::SIMD128f<REGISTER>)
         return ConvertFrom256i_To128f<FROM>(v);
      else if constexpr (CT::SIMD128d<REGISTER>)
         return ConvertFrom256i_To128d<FROM>(v);
      else if constexpr (CT::SIMD128i<REGISTER>)
         return ConvertFrom256i_To128i<TO, FROM>(v);
      else if constexpr (CT::SIMD256f<REGISTER>)
         return ConvertFrom256i_To256f<FROM>(v);
      else if constexpr (CT::SIMD256d<REGISTER>)
         return ConvertFrom256i_To256d<FROM>(v);
      else if constexpr (CT::SIMD256i<REGISTER>)
         return ConvertFrom256i_To256i<TO, FROM>(v);
      else
         LANGULUS_ERROR("Can't convert from __m256i to unsupported");
   }

} // namespace Langulus::SIMD