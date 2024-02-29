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
#include "Store.hpp"
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
   namespace Inner
   {

      /// Convert from one array to another using SIMD                        
      ///   @tparam DEF - default values for elements that are not loaded     
      ///   @tparam TO - type to convert to                                   
      ///   @tparam FROM - type to convert from                               
      ///   @param in - the input data                                        
      ///   @return the resulting register                                    
      template<int DEF, CT::Vector TO, CT::Vector FROM> LANGULUS(INLINED)
      auto Convert(const FROM& in) noexcept {
         using FROM_SIMD = Inner::ToSIMD<FROM, FROM>;
         using TO_SIMD = Inner::ToSIMD<FROM, TO>;
         using D_TO = Decay<TypeOf<TO>>;
         using D_FROM = Decay<TypeOf<FROM>>;

         if constexpr (CT::NotSIMD<FROM_SIMD> or CT::NotSIMD<TO_SIMD>) {
            // FROM can't be wrapped inside a register                  
            return Unsupported {};
         }
         else {
            const FROM_SIMD loaded = Load<DEF>(in);

            if constexpr (CT::Exact<D_FROM, D_TO>)
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

   } // namespace Langulus::SIMD::Inner


   /// Convert numbers, and force output to desired place                     
   ///   @tparam DEF - default values for elements that are not loaded        
   ///   @tparam FROM - source array, scalar, or register (deducible)         
   ///   @tparam TO - the desired element type (deducible)                    
   ///   @return array/scalar                                                 
   template<int DEF, class FROM, class TO> NOD() LANGULUS(INLINED)
   constexpr auto ConvertConstexpr(const FROM& from, TO& to) noexcept {
      using T = Decay<TypeOf<TO>>;

      if constexpr (CT::Vector<FROM>) {
         //                                                             
         // Convert from vectors...                                     
         if constexpr (CT::Scalar<TO>) {
            // ... to a scalar (just use the first element)             
            DenseCast(Inner::GetFirst(to)) = static_cast<T>(DenseCast(Inner::GetFirst(from)));
         }
         else {
            // ... to a vector (convert available elements, default the 
            // rest using DEF)                                          
            constexpr auto S = OverlapCounts<FROM, TO>();
            for (Count i = 0; i < S; ++i)
               DenseCast(to[i]) = static_cast<T>(DenseCast(from[i]));

            constexpr auto SMAX = CountOf<TO>;
            constexpr auto DEFAULT = static_cast<T>(DEF);
            if constexpr (S < SMAX) {
               // Fill the rest with the default value                  
               for (Count i = S; i < SMAX; ++i)
                  DenseCast(to[i]) = DEFAULT;
            }
         }
      }
      else {
         //                                                             
         // Convert from scalars                                        
         const auto scalar = static_cast<T>(DenseCast(Inner::GetFirst(from)));
         for (auto& v : to)
            v = scalar;
      }
   }

   /// Convert numbers, and force output to desired place                     
   ///   @tparam DEF - default values for elements that are not loaded        
   ///   @tparam FROM - source array, scalar, or register (deducible)         
   ///   @tparam TO - the desired element type (deducible)                    
   ///   @attention may generate additional convert/store instructions in     
   ///              order to fit the result in desired output                 
   template<int DEF, class FROM, class TO> LANGULUS(INLINED)
   constexpr void Convert(const FROM& from, TO& to) noexcept {
      using T = Decay<TypeOf<TO>>;

      IF_CONSTEXPR() {
         ConvertConstexpr<DEF>(from, to);
      }
      else if constexpr (CT::Vector<FROM>) {
         //                                                             
         // Convert from vectors...                                     
         if constexpr (CT::Scalar<TO>) {
            // ... to a scalar (just use the first element)             
            DenseCast(Inner::GetFirst(to)) = static_cast<T>(DenseCast(Inner::GetFirst(from)));
         }
         else if constexpr (CT::SIMD<decltype(Inner::Convert<DEF, TO>(from))>) {
            // ... to a vector (convert available elements, default the 
            // rest using DEF) - this is where SIMD steps in            
            Store(Inner::Convert<DEF, TO>(from), to);
         }
         else {
            // Fallback if no SIMD routine available for this conversion
            ConvertConstexpr<DEF>(from, to);
         }
      }
      else {
         //                                                             
         // Convert from scalars                                        
         const auto scalar = static_cast<T>(DenseCast(Inner::GetFirst(from)));
         for (auto& v : to)
            v = scalar;
      }
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
