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
#include "IgnoreWarningsPush.inl"


namespace Langulus::SIMD
{

   /// Fill a register with a single value                                    
   ///   @tparam REGISTER - type of register to fill                          
   ///   @tparam TO - type of data we're filling as                           
   ///   @tparam FROM - type of data we're using to fill (deducible)          
   ///   @param from -  the value to use for filling                          
   ///                  can be inside a vector or an array                    
   ///   @return the filled register                                          
   template<CT::SIMD REGISTER, CT::Vector TO, CT::Scalar FROM>
   NOD() LANGULUS(INLINED)
   decltype(auto) Fill(UNUSED() const FROM& from) noexcept {
      static_assert(CT::NotSIMD<FROM>, "FROM can't be a register");
      using D_TO = Decay<TypeOf<TO>>;
      UNUSED() const auto& value = DenseCast(Inner::GetFirst(from));

   #if LANGULUS_SIMD(128BIT)
      if constexpr (CT::SIMD128i<REGISTER>) {
         if constexpr (CT::SignedInteger8<D_TO>)
            return simde_mm_set1_epi8(static_cast<int8_t>(value));
         else if constexpr (CT::UnsignedInteger8<D_TO>)
            return simde_x_mm_set1_epu8(static_cast<uint8_t>(value));
         else if constexpr (CT::SignedInteger16<D_TO>)
            return simde_mm_set1_epi16(static_cast<int16_t>(value));
         else if constexpr (CT::UnsignedInteger16<D_TO>)
            return simde_x_mm_set1_epu16(static_cast<uint16_t>(value));
         else if constexpr (CT::SignedInteger32<D_TO>)
            return simde_mm_set1_epi32(static_cast<int32_t>(value));
         else if constexpr (CT::UnsignedInteger32<D_TO>)
            return simde_x_mm_set1_epu32(static_cast<uint32_t>(value));
         else if constexpr (CT::SignedInteger64<D_TO>)
            return simde_mm_set1_epi64x(static_cast<int64_t>(value));
         else if constexpr (CT::UnsignedInteger64<D_TO>)
            return simde_x_mm_set1_epu64(static_cast<uint64_t>(value));
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m128i");
      }
      else if constexpr (CT::SIMD128f<REGISTER>) {
         if constexpr (CT::Float<D_TO>) {
            if constexpr (CT::Exact<Decay<decltype(value)>, simde_float32>)
               return simde_mm_broadcast_ss(reinterpret_cast<const simde_float32*>(&value));
            else
               return simde_mm_set1_ps(static_cast<simde_float32>(value));
         }
         else LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m128");
      }
      else if constexpr (CT::SIMD128d<REGISTER>) {
         if constexpr (CT::Double<D_TO>)
            return simde_mm_set1_pd(static_cast<simde_float64>(value));
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m128d");
      }
      else
   #endif

   #if LANGULUS_SIMD(256BIT)
      if constexpr (CT::SIMD256i<REGISTER>) {
         if constexpr (CT::Integer8<D_TO>)
            return simde_mm256_set1_epi8(static_cast<int8_t>(value));
         else if constexpr (CT::Integer16<D_TO>)
            return simde_mm256_set1_epi16(static_cast<int16_t>(value));
         else if constexpr (CT::Integer32<D_TO>)
            return simde_mm256_set1_epi32(static_cast<int32_t>(value));
         else if constexpr (CT::Integer64<D_TO>)
            return simde_mm256_set1_epi64x(static_cast<int64_t>(value));
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m256i");
      }
      else if constexpr (CT::SIMD256f<REGISTER>) {
         if constexpr (CT::Float<D_TO>) {
            if constexpr (CT::Exact<Decay<decltype(value)>, simde_float32>)
               return simde_mm256_broadcast_ss(reinterpret_cast<const simde_float32*>(&value));
            else
               return simde_mm256_set1_ps(static_cast<simde_float32>(value));
         }
         else LANGULUS_ERROR("Unsupported type for SIMD::Fill __m256");
      }
      else if constexpr (CT::SIMD256d<REGISTER>) {
         if constexpr (CT::Double<D_TO>) {
            if constexpr (CT::Exact<Decay<decltype(value)>, simde_float64>)
               return simde_mm256_broadcast_sd(reinterpret_cast<const simde_float64*>(&value));
            else
               return simde_mm256_set1_pd(static_cast<simde_float64>(value));
         }
         else LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m256d");
      }
      else
   #endif

   #if LANGULUS_SIMD(512BIT)
      if constexpr (CT::SIMD512i<REGISTER>) {
         if constexpr (CT::Integer8<D_TO>)
            return simde_mm512_set1_epi8(static_cast<int8_t>(value));
         else if constexpr (CT::Integer16<D_TO>)
            return simde_mm512_set1_epi16(static_cast<int16_t>(value));
         else if constexpr (CT::Integer32<D_TO>)
            return simde_mm512_set1_epi32(static_cast<int32_t>(value));
         else if constexpr (CT::Integer64<D_TO>)
            return simde_mm512_set1_epi64(static_cast<int64_t>(value));
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m512i");
      }
      else if constexpr (CT::SIMD512f<REGISTER>) {
         if constexpr (CT::Float<D_TO>)
            return simde_mm512_set1_ps(static_cast<simde_float32>(value));
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill __m512");
      }
      else if constexpr (CT::SIMD512d<REGISTER>) {
         if constexpr (CT::Double<D_TO>)
            return simde_mm512_set1_pd(static_cast<simde_float64>(value));
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m512d");
      }
      else
   #endif
      LANGULUS_ERROR("Bad REGISTER type for SIMD::Fill");
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
