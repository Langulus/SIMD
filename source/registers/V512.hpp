#pragma once
#include "../Common.hpp"
#include "V256.hpp"


namespace Langulus::SIMD
{

   template<>
   struct V512<simde_float32> {
      LANGULUS(TYPED) simde_float32;
      static constexpr int CTTI_SIMD_Trait = 512;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float32);

      simde__m512 m;

      V512() noexcept = default;

      LANGULUS(INLINED)
      V512(const simde__m512& v) noexcept
         : m {v} {}

      LANGULUS(INLINED)
      static V512 Zero() noexcept {
         return simde_mm512_setzero_ps();
      }
      LANGULUS(INLINED)
      operator simde__m512& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m512 const& () const noexcept {
         return m;
      }
   };

   template<>
   struct V512<simde_float64> {
      LANGULUS(TYPED) simde_float64;
      static constexpr int CTTI_SIMD_Trait = 512;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float64);

      simde__m512d m;

      V512() noexcept = default;

      LANGULUS(INLINED)
      V512(const simde__m512d& v) noexcept
         : m {v} {}

      LANGULUS(INLINED)
      static V512 Zero() noexcept {
         return simde_mm512_setzero_pd();
      }
      LANGULUS(INLINED)
      operator simde__m512d& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m512d const& () const noexcept {
         return m;
      }
   };

   template<IntElement T>
   struct V512<T> {
      LANGULUS(TYPED) T;
      static constexpr int CTTI_SIMD_Trait = 512;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(T);

      simde__m512i m;

      V512() noexcept = default;

      LANGULUS(INLINED)
      V512(const simde__m512i& v) noexcept
         : m {v} {}

      LANGULUS(INLINED)
      static V512 Zero() noexcept {
         return simde_mm512_setzero_si512();
      }
      LANGULUS(INLINED)
      operator simde__m512i& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m512i const& () const noexcept {
         return m;
      }
      
      LANGULUS(INLINED)
      auto UnpackLo() const noexcept {
         if constexpr (CT::SignedInteger8<T>)
            return V512<std::int16_t>  {simde_mm512_unpacklo_epi8 (m, Zero())};
         else if constexpr (CT::UnsignedInteger8<T>)
            return V512<std::uint16_t> {simde_mm512_unpacklo_epi8 (m, Zero())};
         else if constexpr (CT::SignedInteger16<T>)
            return V512<std::int32_t>  {simde_mm512_unpacklo_epi16(m, Zero())};
         else if constexpr (CT::UnsignedInteger16<T>)
            return V512<std::uint32_t> {simde_mm512_unpacklo_epi16(m, Zero())};
         else if constexpr (CT::SignedInteger32<T>)
            return V512<std::int64_t>  {simde_mm512_unpacklo_epi32(m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V512<std::uint64_t> {simde_mm512_unpacklo_epi32(m, Zero())};
         else
            static_assert(false, "Can't unpack this type");
      }

      LANGULUS(INLINED)
      auto UnpackHi() const noexcept {
         if constexpr (CT::SignedInteger8<T>)
            return V512<std::int16_t>  {simde_mm512_unpackhi_epi8 (m, Zero())};
         else if constexpr (CT::UnsignedInteger8<T>)
            return V512<std::uint16_t> {simde_mm512_unpackhi_epi8 (m, Zero())};
         else if constexpr (CT::SignedInteger16<T>)
            return V512<std::int32_t>  {simde_mm512_unpackhi_epi16(m, Zero())};
         else if constexpr (CT::UnsignedInteger16<T>)
            return V512<std::uint32_t> {simde_mm512_unpackhi_epi16(m, Zero())};
         else if constexpr (CT::SignedInteger32<T>)
            return V512<std::int64_t>  {simde_mm512_unpackhi_epi32(m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V512<std::uint64_t> {simde_mm512_unpackhi_epi32(m, Zero())};
         else
            static_assert(false, "Can't unpack this type");
      }

      LANGULUS(INLINED)
      auto Pack() const noexcept {
         if constexpr (CT::Integer8<T>)
            return *this;
         else if constexpr (CT::SignedInteger16<T>) {
            const auto lo_lane = simde_mm512_castsi512_si256(m);
            const auto hi_lane = simde_mm512_extracti256_si512(m, 1);
            return V256<std::int8_t> {
               simde_mm256_packs_epi16(lo_lane, hi_lane)
            }.Pack();
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            const auto lo_lane = simde_mm512_castsi512_si256(m);
            const auto hi_lane = simde_mm512_extracti256_si512(m, 1);
            return V256<std::uint8_t> {
               simde_mm256_packus_epi16(lo_lane, hi_lane)
            }.Pack();
         }
         else if constexpr (CT::SignedInteger32<T>)
            return V512<std::int16_t>  {simde_mm512_packs_epi32 (m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V512<std::uint16_t> {simde_mm512_packus_epi32(m, Zero())};
         else if constexpr (CT::SignedInteger64<T>)
            return V256<std::int32_t>  {simde_mm512_cvtepi64_epi32(m)};
         else if constexpr (CT::UnsignedInteger64<T>)
            return V256<std::uint32_t> {simde_mm512_cvtepi64_epi32(m)};
         else
            static_assert(false, "Can't pack this type");
      }
   };

} // namespace Langulus::SIMD

namespace Langulus::CT
{

   /// Concept for 512bit SIMD float registers                                
   template<class...T>
   concept SIMD512f = ((Deref<T>::CTTI_SIMD_Trait == 512
       and CT::Float<TypeOf<T>>) and ...);

   /// Concept for 512bit SIMD double registers                               
   template<class...T>
   concept SIMD512d = ((Deref<T>::CTTI_SIMD_Trait == 512
       and CT::Double<TypeOf<T>>) and ...);

   /// Concept for 512bit SIMD integer/bool registers                         
   template<class...T>
   concept SIMD512i = ((Deref<T>::CTTI_SIMD_Trait == 512
       and CT::Integer<TypeOf<T>>) and ...);

   /// Concept for 512bit SIMD registers                                      
   template<class...T>
   concept SIMD512  = ((Deref<T>::CTTI_SIMD_Trait == 512) and ...);

} // namespace Langulus::CT