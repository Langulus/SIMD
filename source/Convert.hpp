///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "Load.hpp"
#include "Store.hpp"

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

      /// Used to detect missing SIMD routine                                 
      template<Element> NOD() LANGULUS(INLINED)
      constexpr Unsupported ConvertSIMD(CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Convert from one register to another                                
      ///   @tparam TO - type of element to convert to                        
      ///   @param in - register to convert from                              
      ///   @return the resulting register, or Unsupported if not possible    
      template<Element TO> NOD() LANGULUS(INLINED)
      auto ConvertSIMD(CT::SIMD auto in) noexcept {
         using R = decltype(in);
         using T = TypeOf<R>;

         if constexpr (CT::Similar<T, TO>) {
            // No conversion required, just forward the register        
            return in;
         }
         else if constexpr (CT::SIMD128<R>) {
            if      constexpr (CT::Float<T>)    return ConvertFrom128f<TO>(in);
            else if constexpr (CT::Double<T>)   return ConvertFrom128d<TO>(in);
            else if constexpr (CT::Integer<T>)  return ConvertFrom128i<TO>(in);
         }
         else if constexpr (CT::SIMD256<R>) {
            if      constexpr (CT::Float<T>)    return ConvertFrom256f<TO>(in);
            else if constexpr (CT::Double<T>)   return ConvertFrom256d<TO>(in);
            else if constexpr (CT::Integer<T>)  return ConvertFrom256i<TO>(in);
         }
         else if constexpr (CT::SIMD512<R>) {
            if      constexpr (CT::Float<T>)    return ConvertFrom512f<TO>(in);
            else if constexpr (CT::Double<T>)   return ConvertFrom512d<TO>(in);
            else if constexpr (CT::Integer<T>)  return ConvertFrom512i<TO>(in);
         }
         else static_assert(false, "Can't convert from unsupported");
      }

      /// Convert scalars/arrays at compile-time, if possible                 
      ///   @tparam TO - the desired element type                             
      ///   @param in - scalar/vector to convert from                         
      ///   @return std::array or scalar, depending on the input              
      template<Element TO> NOD() LANGULUS(INLINED)
      constexpr auto ConvertConstexpr(const CT::NotSIMD auto& in) noexcept {
         using FROM = Deref<decltype(in)>;

         if constexpr (CT::Vector<FROM>) {
            // Convert from vectors                                     
            ::std::array<TO, CountOf<FROM>> result;
            for (Count i = 0; i < CountOf<FROM>; ++i)
               result[i] = static_cast<TO>(in[i]);
            return result;
         }
         else {
            // Convert from scalar                                      
            return static_cast<TO>(in);
         }
      }

      /// Convert scalars/arrays/registers and return a register, if possible 
      ///   @tparam DEF - default value for setting elements outside array,   
      ///      used only if input array is smaller than chosen register       
      ///   @tparam TO - the desired element type                             
      ///   @param in - scalar/vector/register to convert from                
      ///   @return scalar/vector/register/unsupported                        
      template<auto DEF, Element TO> NOD() LANGULUS(INLINED)
      auto Convert(const auto& in) noexcept {
         using FROM = Deref<decltype(in)>;

         if constexpr (CT::SIMD<FROM>) {
            // Input is already a register, skip loading                
            return ConvertSIMD<TO>(in);
         }
         else if constexpr (CT::Vector<FROM>) {
            // Convert from vectors                                     
            // Attempt loading input array into a register              
            const auto v = Load<DEF>(in);

            if constexpr (CT::Unsupported<decltype(v)>) {
               // Load to register fails, fallback                      
               return ConvertConstexpr<TO>(in);
            }
            else {
               // Load was a success, now test if SIMD conversion is    
               // supported                                             
               const auto converted = ConvertSIMD<TO>(v);
               if constexpr (CT::Unsupported<decltype(converted)>) {
                  // SIMD conversion fails, fallback                    
                  return ConvertConstexpr<TO>(in);
               }
               else {
                  static_assert(CT::SIMD<decltype(converted)>,
                     "Conversion result isn't a V type, "
                     "did you forget return R {...}?");
                  return converted;
               }
            }
         }
         else {
            // Convert from scalar                                      
            return static_cast<TO>(in);
         }
      }

   } // namespace Langulus::SIMD::Inner

   /// Convert numbers, and force output to desired place                     
   ///   @tparam DEF - default value for setting elements outside array,      
   ///      used only if input array is smaller than chosen register          
   ///   @tparam OUT - the desired scalar/array/register location (deducible) 
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in 'out'. Use Inner::Convert if you    
   ///      don't want this.                                                  
   template<auto DEF, CT::NoIntent OUT> LANGULUS(INLINED)
   constexpr void Convert(const auto& val, OUT& out) noexcept {
      using TO = TypeOf<OUT>;

      IF_CONSTEXPR() {
         // Converting in a contexpr context                            
         Store(Inner::ConvertConstexpr<TO>(DeintCast(val)), out);
      }
      else {
         // Converting using SIMD, hopefully                            
         if constexpr (CT::SIMD<OUT>)
            out = Inner::Convert<DEF, TO>(DeintCast(val));
         else
            Store(Inner::Convert<DEF, TO>(DeintCast(val)), out);
      }
   }

   /// Convert numbers                                                        
   ///   @tparam VAL - array, scalar, or register (deducible)                 
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @attention will generate additional store (and convert) instructions 
   ///      in order to fit the result in an instance of 'OUT'. Use           
   ///      Inner::Convert if you don't want this.                            
   template<class VAL, CT::NoIntent OUT = LosslessArray<VAL, VAL>>
   NOD() LANGULUS(INLINED)
   constexpr OUT Convert(const VAL& val) noexcept {
      OUT out;
      Convert(DeintCast(val), out);
      return out;
   }

} // namespace Langulus::SIMD
