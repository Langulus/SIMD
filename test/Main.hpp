///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include <SIMD/SIMD.hpp>
#include <cstdint>
#include <cstddef>
#include <random>
#include <bitset>

using namespace Langulus;

//#define LANGULUS_STD_BENCHMARK

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#define NUMBERS_REAL() \
   float, double

#define NUMBERS_SIGNED() \
   ::std::int8_t,   \
   ::std::int16_t,  \
   ::std::int32_t,  \
   ::std::int64_t,  \
   NUMBERS_REAL()

#define NUMBERS_UNSIGNED() \
   ::std::uint8_t,  \
   ::std::uint16_t, \
   ::std::uint32_t, \
   ::std::uint64_t, \
   Byte,            \
   char8_t,         \
   char16_t,        \
   char32_t,        \
   wchar_t

#define NUMBERS_INT() \
   ::std::int8_t,   \
   ::std::int16_t,  \
   ::std::int32_t,  \
   ::std::int64_t,  \
   ::std::uint8_t,  \
   ::std::uint16_t, \
   ::std::uint32_t, \
   ::std::uint64_t, \
   Byte,            \
   char8_t,         \
   char16_t,        \
   char32_t,        \
   wchar_t

#define NUMBERS_ALL() \
   NUMBERS_SIGNED(),  \
   NUMBERS_UNSIGNED()

#define VECTORS_REAL(S)         \
   (Vector<float, S>),          \
   (Vector<double, S>)

#define VECTORS_SIGNED(S)       \
   (Vector<::std::int8_t, S>),  \
   (Vector<::std::int16_t, S>), \
   (Vector<::std::int32_t, S>), \
   (Vector<::std::int64_t, S>), \
   VECTORS_REAL(S)

#define VECTORS_UNSIGNED(S)     \
   (Vector<::std::uint8_t, S>), \
   (Vector<::std::uint16_t, S>),\
   (Vector<::std::uint32_t, S>),\
   (Vector<::std::uint64_t, S>),\
   (Vector<Byte, S>),           \
   (Vector<char8_t, S>),        \
   (Vector<char16_t, S>),       \
   (Vector<char32_t, S>),       \
   (Vector<wchar_t, S>)

#define VECTORS_INT(S) \
   (Vector<::std::int8_t, S>),  \
   (Vector<::std::int16_t, S>), \
   (Vector<::std::int32_t, S>), \
   (Vector<::std::int64_t, S>), \
   (Vector<::std::uint8_t, S>), \
   (Vector<::std::uint16_t, S>),\
   (Vector<::std::uint32_t, S>),\
   (Vector<::std::uint64_t, S>),\
   (Vector<Byte, S>),           \
   (Vector<char8_t, S>),        \
   (Vector<char16_t, S>),       \
   (Vector<char32_t, S>),       \
   (Vector<wchar_t, S>)

#define VECTORS_ALL(S) \
   VECTORS_SIGNED(S),  \
   VECTORS_UNSIGNED(S)

using uint = unsigned int;
template<class T>
using some = std::vector<T>;

template<class T, class A>
void InitOne(T& a, A&& b) noexcept {
   if constexpr (CT::Sparse<T>) {
      using DT = Decay<T>;
      if (a)
         delete a;
      a = new DT {static_cast<DT>(b)};
   }
   else a = static_cast<Decay<TypeOf<T>>>(b);
}

///                                                                           
/// Satisfies the CT::Vector concept when C > 1                               
/// Satisfied the CT::Scalar concept when C == 1                              
///                                                                           
#pragma pack(push, 1)
template<CT::Dense T, Count C>
struct Vector {
   LANGULUS(TYPED) T;
   static constexpr Count MemberCount = C;

   T mArray[C];

   struct iterator {
      const T* marker;

      iterator() = delete;
      iterator(const T* a) : marker {a} {}

      bool operator == (const iterator& it) const noexcept {
         return marker == it.marker;
      }

      // Prefix operator                                                
      iterator& operator ++ () noexcept {
         ++marker;
         return *this;
      }

      // Suffix operator                                                
      iterator operator ++ (int) noexcept {
         const auto backup = *this;
         operator ++ ();
         return backup;
      }

      decltype(auto) operator * () const noexcept {
         return DenseCast(*marker);
      }
   };

   auto begin() const noexcept {
      return iterator {mArray};
   }

   auto end() const noexcept {
      return iterator {mArray + C};
   }

   Vector() {
      static std::random_device rd;
      static std::mt19937 gen(rd());

      for (auto& i : mArray) {
         i = static_cast<T>(gen() % 66);
         if (i == T {0})
            i = T {1};
      }
   }
   
   template<class ALT>
   constexpr Vector(const std::array<ALT, C>& v) {
      for (Count i = 0; i < C; ++i)
         mArray[i] = static_cast<T>(v[i]);
   }

   constexpr Vector(const Vector& v) {
      for (Count i = 0; i < C; ++i)
         mArray[i] = v.mArray[i];
   }

   constexpr Vector(const Decay<T>& s) {
      for (Count i = 0; i < C; ++i)
         mArray[i] = s;
   }

   constexpr bool operator == (const Vector& e) const noexcept {
      for (Count i = 0; i < C; ++i)
         if (DenseCast(mArray[i]) != DenseCast(e.mArray[i]))
            return false;
      return true;
   }

   constexpr Vector& operator = (const Vector& b) noexcept {
      for (Count i = 0; i < C; ++i)
         DenseCast(mArray[i]) = DenseCast(b.mArray[i]);
      return *this;
   }

   constexpr Vector& operator = (const Decay<T>& b) noexcept {
      for (Count i = 0; i < C; ++i)
         DenseCast(mArray[i]) = b;
      return *this;
   }

   constexpr explicit operator T& () const noexcept requires (C==1) {
      return const_cast<T&>(mArray[0]);
   }

   constexpr const T& operator [](auto i) const noexcept {
      return mArray[i];
   }

   constexpr T& operator [](auto i) noexcept {
      return mArray[i];
   }
};
#pragma pack(pop)
