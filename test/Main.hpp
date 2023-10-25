///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
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

#define NUMBERS_SIGNED() \
   ::std::int8_t,    ::std::int8_t*,  \
   ::std::int16_t,   ::std::int16_t*, \
   ::std::int32_t,   ::std::int32_t*, \
   ::std::int64_t,   ::std::int64_t*, \
   float, float*, \
   double, double*

#define NUMBERS_UNSIGNED() \
   ::std::uint8_t,   ::std::uint8_t*,  \
   ::std::uint16_t,  ::std::uint16_t*, \
   ::std::uint32_t,  ::std::uint32_t*, \
   ::std::uint64_t,  ::std::uint64_t*, \
   Byte, Byte*, \
   char8_t, char8_t*, \
   char16_t, char16_t*, \
   char32_t, char32_t*, \
   wchar_t, wchar_t*

#define NUMBERS_INT() \
   ::std::int8_t,    ::std::int8_t*,  \
   ::std::int16_t,   ::std::int16_t*, \
   ::std::int32_t,   ::std::int32_t*, \
   ::std::int64_t,   ::std::int64_t*, \
   ::std::uint8_t,   ::std::uint8_t*,  \
   ::std::uint16_t,  ::std::uint16_t*, \
   ::std::uint32_t,  ::std::uint32_t*, \
   ::std::uint64_t,  ::std::uint64_t*, \
   Byte, Byte*, \
   char8_t, char8_t*, \
   char16_t, char16_t*, \
   char32_t, char32_t*, \
   wchar_t, wchar_t*

#define NUMBERS_ALL() \
   NUMBERS_SIGNED(), \
   NUMBERS_UNSIGNED()

#define VECTORS_SIGNED(S) \
   (Vector<::std::int8_t, S>),   (Vector<::std::int8_t*, S>), \
   (Vector<::std::int16_t, S>),  (Vector<::std::int16_t*, S>), \
   (Vector<::std::int32_t, S>),  (Vector<::std::int32_t*, S>), \
   (Vector<::std::int64_t, S>),  (Vector<::std::int64_t*, S>), \
   (Vector<float, S>),           (Vector<float*, S>), \
   (Vector<double, S>),          (Vector<double*, S>)

#define VECTORS_UNSIGNED(S) \
   (Vector<::std::uint8_t, S>),  (Vector<::std::uint8_t*, S>), \
   (Vector<::std::uint16_t, S>), (Vector<::std::uint16_t*, S>), \
   (Vector<::std::uint32_t, S>), (Vector<::std::uint32_t*, S>), \
   (Vector<::std::uint64_t, S>), (Vector<::std::uint64_t*, S>), \
   (Vector<Byte, S>),            (Vector<Byte*, S>), \
   (Vector<char8_t, S>),         (Vector<char8_t*, S>), \
   (Vector<char16_t, S>),        (Vector<char16_t*, S>), \
   (Vector<char32_t, S>),        (Vector<char32_t*, S>), \
   (Vector<wchar_t, S>),         (Vector<wchar_t*, S>)

#define VECTORS_INT(S) \
   (Vector<::std::int8_t, S>),   (Vector<::std::int8_t*, S>), \
   (Vector<::std::int16_t, S>),  (Vector<::std::int16_t*, S>), \
   (Vector<::std::int32_t, S>),  (Vector<::std::int32_t*, S>), \
   (Vector<::std::int64_t, S>),  (Vector<::std::int64_t*, S>), \
   (Vector<::std::uint8_t, S>),  (Vector<::std::uint8_t*, S>), \
   (Vector<::std::uint16_t, S>), (Vector<::std::uint16_t*, S>), \
   (Vector<::std::uint32_t, S>), (Vector<::std::uint32_t*, S>), \
   (Vector<::std::uint64_t, S>), (Vector<::std::uint64_t*, S>), \
   (Vector<Byte, S>),            (Vector<Byte*, S>), \
   (Vector<char8_t, S>),         (Vector<char8_t*, S>), \
   (Vector<char16_t, S>),        (Vector<char16_t*, S>), \
   (Vector<char32_t, S>),        (Vector<char32_t*, S>), \
   (Vector<wchar_t, S>),         (Vector<wchar_t*, S>)

#define VECTORS_ALL(S) \
   VECTORS_SIGNED(S), \
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

/// Satisfies the CT::Vector concept                                          
#pragma pack(push, 1)
template<class T, Count C>
struct /*alignas(Langulus::Alignment)*/ Vector {
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
         if constexpr (CT::Sparse<T>) {
            using TD = Decay<T>;
            i = new TD {static_cast<TD>(gen() % 66)};
            if (*i == TD {0})
               *i = TD {1};
         }
         else {
            i = static_cast<T>(gen() % 66);
            if (i == T {0})
               i = T {1};
         }
      }
   }
   
   Vector(const Vector& v) {
      for (Count i = 0; i < C; ++i) {
         if constexpr (CT::Sparse<T>)
            mArray[i] = new Decay<T> {*v.mArray[i]};
         else
            mArray[i] = v.mArray[i];
      }
   }

   Vector(const Decay<T>& s) {
      for (Count i = 0; i < C; ++i) {
         if constexpr (CT::Sparse<T>)
            mArray[i] = new Decay<T> {s};
         else
            mArray[i] = s;
      }
   }

   ~Vector() {
      for (auto& i : mArray) {
         if constexpr (CT::Sparse<T>)
            delete i;
      }
   }

   bool operator == (const Vector& e) const noexcept {
      for (Count i = 0; i < C; ++i)
         if (DenseCast(mArray[i]) != DenseCast(e.mArray[i]))
            return false;
      return true;
   }

   Vector& operator = (const Vector& b) noexcept {
      for (Count i = 0; i < C; ++i)
         DenseCast(mArray[i]) = DenseCast(b.mArray[i]);
      return *this;
   }

   Vector& operator = (const Decay<T>& b) noexcept {
      for (Count i = 0; i < C; ++i)
         DenseCast(mArray[i]) = b;
      return *this;
   }

   explicit operator const T& () const noexcept requires (C==1) {
      return mArray[0];
   }
   explicit operator T& () noexcept requires (C==1) {
      return mArray[0];
   }

   const T& operator [](auto i) const noexcept {
      return mArray[i];
   }

   T& operator [](auto i) noexcept {
      return mArray[i];
   }
};
#pragma pack(pop)
