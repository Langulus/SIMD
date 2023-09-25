///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Load.hpp"
#include "IgnoreWarningsPush.inl"

#if LANGULUS_SIMD(128BIT)
   #include "converters/From128f.hpp"
   #include "converters/From128d.hpp"
   #include "converters/From128i.hpp"
#endif
#if LANGULUS_SIMD(256BIT)
   #include "converters/From256f.hpp"
   #include "converters/From256d.hpp"
   #include "converters/From256i.hpp"
#endif
#if LANGULUS_SIMD(512BIT)
   #include "converters/From512f.hpp"
   #include "converters/From512d.hpp"
   #include "converters/From512i.hpp"
#endif


namespace Langulus::SIMD
{

   /// Convert from one array to another using SIMD                           
   ///   @tparam DEF - default values for elements that are not loaded        
   ///   @tparam TO - type to convert to                                      
   ///   @tparam FROM - type to convert from                                  
   ///   @param in - the input data                                           
   ///   @return the resulting register                                       
   template<int DEF, class TO, class FROM>
   LANGULUS(INLINED)
   auto Convert(const FROM& in) noexcept {
      using FROM_SIMD = Inner::ToSIMD<FROM, FROM>;
      using TO_SIMD = Inner::ToSIMD<FROM, TO>;
      using D_TO = Decay<TO>;
      using D_FROM = Decay<FROM>;

      if constexpr (CT::NotSIMD<FROM_SIMD> or CT::NotSIMD<TO_SIMD>)
         // FROM can't be wrapped inside a register                     
         return Unsupported {};
      else {
         const FROM_SIMD loaded = Load<DEF>(in);
         if constexpr (CT::Exact<TypeOf<D_FROM>, D_TO>)
            // Early exit if Load was enough                            
            return loaded;

         #if LANGULUS_SIMD(128BIT)
            else if constexpr (CT::SIMD128f<FROM_SIMD>)
               return Inner::ConvertFrom128f<D_TO, TO_SIMD>(loaded);
            else if constexpr (CT::SIMD128d<FROM_SIMD>)
               return Inner::ConvertFrom128d<D_TO, TO_SIMD>(loaded);
            else if constexpr (CT::SIMD128i<FROM_SIMD>)
               return Inner::ConvertFrom128i<D_TO, D_FROM, TO_SIMD>(loaded);
         #endif

         #if LANGULUS_SIMD(256BIT)
            else if constexpr (CT::SIMD256f<FROM_SIMD>)
               return Inner::ConvertFrom256f<D_TO, TO_SIMD>(loaded);
            else if constexpr (CT::SIMD256d<FROM_SIMD>)
               return Inner::ConvertFrom256d<D_TO, TO_SIMD>(loaded);
            else if constexpr (CT::SIMD256i<FROM_SIMD>)
               return Inner::ConvertFrom256i<D_TO, D_FROM, TO_SIMD>(loaded);
         #endif

         #if LANGULUS_SIMD(512BIT)
            else if constexpr (CT::SIMD512f<FROM_SIMD>)
               return Inner::ConvertFrom512f<D_TO, TO_SIMD>(loaded);
            else if constexpr (CT::SIMD512d<FROM_SIMD>)
               return Inner::ConvertFrom512d<D_TO, TO_SIMD>(loaded);
            else if constexpr (CT::SIMD512i<FROM_SIMD>)
               return Inner::ConvertFrom512i<D_TO, D_FROM, TO_SIMD>(loaded);
         #endif

         else LANGULUS_ERROR("Can't convert from unsupported");
      }
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
