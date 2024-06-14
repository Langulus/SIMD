///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           

/// INTENTIONALLY NOT GUARDED                                                 
/// Include this file once in each cpp file, after all other headers          
#ifdef TWOBLUECUBES_SINGLE_INCLUDE_CATCH_HPP_INCLUDED
   #error Catch has been included prior to this header
#endif

//#define LANGULUS_STD_BENCHMARK

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "Main.hpp"
#include <catch2/catch.hpp>


/// See https://github.com/catchorg/Catch2/blob/devel/docs/tostring.md        
CATCH_TRANSLATE_EXCEPTION(::Langulus::Exception const& ex) {
   return fmt::format("{}", ex);
}

namespace Catch {
   template<>
   struct StringMaker<char8_t> {
      static std::string convert(char8_t const& value) {
         return std::to_string(static_cast<int>(value));
      }
   };

   template<>
   struct StringMaker<char16_t> {
      static std::string convert(char16_t const& value) {
         return std::to_string(static_cast<int>(value));
      }
   };

   template<>
   struct StringMaker<wchar_t> {
      static std::string convert(wchar_t const& value) {
         return std::to_string(static_cast<int>(value));
      }
   };

   template<>
   struct StringMaker<::Langulus::Byte> {
      static std::string convert(::Langulus::Byte const& value) {
         return std::to_string(static_cast<int>(value.mValue));
      }
   };
}

using timer = Catch::Benchmark::Chronometer;

template<class T>
using uninitialized = Catch::Benchmark::storage_for<T>;