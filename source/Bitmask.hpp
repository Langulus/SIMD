///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Common.hpp"


namespace Langulus::SIMD
{

   /// Bitmask type, used as result from comparison SIMD operations           
   /// Each comparison operation maps exactly to one bit in this mask         
   /// Internal type representation is designed to be directly mappable to    
   /// _mm_movemask_epi8/_mm512_cmpeq_epi8_mask instrinsic results, without   
   /// any implicit promotions or truncations                                 
   ///   @tparam C - number of bits in the bitmask                            
   template<Count C>
   struct Bitmask {
      static_assert(C <= 64 and C > 0, "C must be in the range [1:64]");

      static constexpr Count MemberCount = C;
      static constexpr bool IsBitmask = true;
      using Type = Conditional<C <= 32, ::std::int32_t, ::std::int64_t>;

      static constexpr Type GetMask() noexcept {
         if constexpr (C == 32)
            return 0xFFFFFFFF;
         else if constexpr (C == 64)
            return 0xFFFFFFFFFFFFFFFF;
         else
            return (Type {1} << C) - Type {1};
      }

      static constexpr Type Mask = GetMask();

      Type mValue {};

      constexpr Bitmask() noexcept = default;
      constexpr Bitmask(const Bitmask&) noexcept = default;
      constexpr Bitmask(Bitmask&&) noexcept = default;
      constexpr explicit Bitmask(const Type& v) noexcept
         : mValue {v & Mask} {}

      struct iterator {
         Offset marker;
         const Type& bitset;

         iterator() = delete;
         iterator(const Offset a, const Type& set)
            : marker {a}
            , bitset {set} {}

         NOD() constexpr bool operator == (const iterator& it) const noexcept {
            return marker == it.marker;
         }

         // Prefix operator                                             
         constexpr iterator& operator ++ () noexcept {
            ++marker;
            return *this;
         }

         // Suffix operator                                             
         NOD() constexpr iterator operator ++ (int) noexcept {
            const auto backup = *this;
            operator ++ ();
            return backup;
         }

         NOD() constexpr bool operator * () const noexcept {
            return 0 != (bitset & (Type {1} << marker));
         }
      };

      NOD() auto begin() const noexcept {
         return iterator {0, mValue};
      }

      NOD() auto end() const noexcept {
         return iterator {C, mValue};
      }

      /// Implicit bool operator                                              
      ///   @return true if all bits in Mask are set                          
      constexpr operator bool() const noexcept {
         return mValue == Mask;
      }

      Bitmask& operator = (const Bitmask&) noexcept = default;
      Bitmask& operator = (const ::std::array<bool, C>& a) noexcept {
         for (Type i = 0; i < Type {C}; ++i)
            mValue |= (static_cast<Type>(a[i]) << i);
         return *this;
      }
      Bitmask& operator = (bool a) noexcept requires (C == 1) {
         mValue = a;
         return *this;
      }

      NOD() constexpr bool operator == (const Bitmask& a) const noexcept {
         return mValue == a.mValue;
      }

      Bitmask& operator |= (const Type& a) noexcept {
         mValue |= a;
         return *this;
      }

      Bitmask& operator &= (const Type& a) noexcept {
         mValue &= a;
         return *this;
      }

      Bitmask& operator ^= (const Type& a) noexcept {
         mValue ^= a;
         return *this;
      }

      NOD() constexpr bool operator [] (const Offset& idx) const noexcept {
         LANGULUS_ASSUME(UserAssumes, idx < C, "Index out of limits");
         return 0 != (mValue & (Type {1} << idx));
      }

      struct BitSwitcher {
         Bitmask<C>& mOwner;
         const Type mTag;

         constexpr BitSwitcher& operator = (const bool flag) noexcept {
            if (flag)
               mOwner |= mTag;
            else
               mOwner &= ~mTag;
            return *this;
         }
      };

      NOD() constexpr BitSwitcher operator [] (const Offset& idx) noexcept {
         LANGULUS_ASSUME(UserAssumes, idx < C, "Index out of limits");
         return BitSwitcher {*this, Type {1} << idx};
      }
   };

} // namespace Langulus::SIMD