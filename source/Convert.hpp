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
      template<Element> LANGULUS(INLINED)
      constexpr Unsupported ConvertSIMD(CT::NotSIMD auto) noexcept {
         return {};
      }

      /// Convert from one register to another                                
      ///   @tparam TO - type of element to convert to                        
      ///   @param in - register to convert from                              
      ///   @return the resulting register, or Unsupported if not possible    
      ///   @attention this function doesn't guarantee that all elements in   
      ///      'in' will be converted - only the amount that fits in the      
      ///      biggest available hardware register                            
      template<Element TO> LANGULUS(INLINED)
      auto ConvertSIMD(CT::SIMD auto in) noexcept {
         using R = decltype(in);
         using T = TypeOf<R>;

         if constexpr (CT::Similar<T, TO>) {
            LANGULUS_SIMD_VERBOSE("No conversion required");
            return in;
         }
         else
         #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<R>) {
            if      constexpr (CT::Float<T>)    return ConvertFrom128f<TO>(in);
            else if constexpr (CT::Double<T>)   return ConvertFrom128d<TO>(in);
            else if constexpr (CT::Integer<T>)  return ConvertFrom128i<TO>(in);
         }
         else
         #endif
         #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<R>) {
            if      constexpr (CT::Float<T>)    return ConvertFrom256f<TO>(in);
            else if constexpr (CT::Double<T>)   return ConvertFrom256d<TO>(in);
            else if constexpr (CT::Integer<T>)  return ConvertFrom256i<TO>(in);
         }
         else
         #endif
         #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<R>) {
            if      constexpr (CT::Float<T>)    return ConvertFrom512f<TO>(in);
            else if constexpr (CT::Double<T>)   return ConvertFrom512d<TO>(in);
            else if constexpr (CT::Integer<T>)  return ConvertFrom512i<TO>(in);
         }
         else
         #endif
         static_assert(false, "Can't convert from unsupported");
      }

      /// Convert scalars/arrays at compile-time, if possible                 
      ///   @tparam TO - the desired element type                             
      ///   @param in - scalar/vector to convert from                         
      ///   @return std::array or scalar, depending on the input              
      template<Element TO> LANGULUS(INLINED)
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
            return static_cast<TO>(DenseCast(in));
         }
      }

      /// Convert scalars/arrays/registers and return a register, if possible 
      ///   @tparam DEF - default value for setting elements outside array,   
      ///      used only if input array is smaller than chosen register       
      ///   @tparam TO - the desired element type                             
      ///   @param in - scalar/vector/register to convert from                
      ///   @return scalar/vector/register/unsupported                        
      ///   @attention this function doesn't guarantee that all elements in   
      ///      'in' will be converted when ConvertSIMD is used - only the     
      ///      amount that fits in the biggest available hardware register    
      template<auto DEF, Element TO> LANGULUS(INLINED)
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
            return static_cast<TO>(DenseCast(in));
         }
      }

   } // namespace Langulus::SIMD::Inner

   /// Convert scalar/array/register, and force output to desired place       
   ///   @tparam DEF - default value for setting elements outside array,      
   ///      used only if input extent is smaller than output extent           
   ///   @param val - what scalar/array/register are we converting?           
   ///   @param out - what scalar/array/register are we converting into?      
   template<auto DEF, class INPUT, CT::NoIntent OUTPUT> LANGULUS(INLINED)
   constexpr void Convert(const INPUT& val, OUTPUT& out) noexcept {
      using FROM = TypeOf<Deint<INPUT>>;
      using TO   = TypeOf<OUTPUT>;

      if constexpr (CT::Vector<OUTPUT>) {
         if (::std::is_constant_evaluated()) {
            // Converting in a contexpr context                         
            Store(Inner::ConvertConstexpr<TO>(DeintCast(val)), out);
         }
         else {
            // Converting using SIMD, hopefully                         
            using CONVERTED_TYPE = decltype(Inner::Convert<DEF, TO>(DeintCast(val)));
            constexpr bool supported = CT::SIMD<CONVERTED_TYPE>;
            constexpr auto CI = CountOf<CONVERTED_TYPE>;
            constexpr auto CO = CountOf<OUTPUT>;

            if constexpr (not supported) {
               // Will always utilize the fallback converter            
               Store(Inner::ConvertConstexpr<TO>(DeintCast(val)), out);
            }
            else if constexpr (CI >= CO or CI == 1) {
               // We're able to do the conversion with a single register
               // (or SIMD is not required at all)                      
               if constexpr (CT::SIMD<OUTPUT>)
                  out = Inner::Convert<DEF, TO>(DeintCast(val));
               else
                  Store(Inner::Convert<DEF, TO>(DeintCast(val)), out);
            }
            else {
               // We have to divide the conversion into multiple regs   
               // This happens when we convert float[4] to double[4]    
               // without AVX support for example                       
               constexpr Count left = Roof2(CountOf<INPUT>/2);
               static_assert(left < CountOf<INPUT>,
                  "Can't properly split the input vector");
               static_assert(left < CountOf<OUTPUT>,
                  "Can't properly split the output vector");

               using LEFTI  = const FROM(&)[left];
               using RIGHTI = const FROM(&)[CountOf<INPUT>  - left];
               using LEFTO  = TO(&)[left];
               using RIGHTO = TO(&)[CountOf<OUTPUT> - left];

               // Nest the two parts so that splitting can occur        
               // statically multiple times if it has to                
               auto input = reinterpret_cast<FROM const*>(SparseCast(DeintCast(val)));

               if constexpr (CT::SIMD<OUTPUT>) {
                  LosslessRegister<LEFTO> out1;
                  LosslessRegister<RIGHTO> out2;
                  Convert<DEF>(reinterpret_cast<LEFTI> (input[0]), out1);
                  Convert<DEF>(reinterpret_cast<RIGHTI>(input[left]), out2);
                  ConcatSIMD(out1, out2, out);
               }
               else {
                  auto output = reinterpret_cast<TO*>(SparseCast(out));
                  Convert<DEF>(
                     reinterpret_cast<LEFTI>(input[0]),
                     reinterpret_cast<LEFTO>(output[0])
                  );
                  Convert<DEF>(
                     reinterpret_cast<RIGHTI>(input[left]),
                     reinterpret_cast<RIGHTO>(output[left])
                  );
               }
            }
         }
      }
      else GetFirst(out) = static_cast<TO>(GetFirst(DeintCast(val)));
   }

   /// Convert scalar/array/register                                          
   ///   @tparam OUT - the desired output type (lossless array by default)    
   ///   @param val - what scalar/array/register are we converting?           
   template<class VAL, CT::NoIntent OUT = LosslessArray<VAL, VAL>> LANGULUS(INLINED)
   constexpr OUT Convert(const VAL& val) noexcept {
      OUT out;
      Convert(DeintCast(val), out);
      return out;
   }

} // namespace Langulus::SIMD
