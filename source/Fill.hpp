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
   template<CT::SIMD REGISTER, class TO, CT::Scalar FROM>
   NOD() LANGULUS(INLINED)
   decltype(auto) Fill(const FROM& from) noexcept {
      static_assert(CT::NotSIMD<FROM>,  "FROM can't be a register");
      static_assert(CT::Decayed<TO>,    "TO must be simplified at this point");
      static_assert(CountOf<FROM> == 1, "Filling uses only the first element");
      using OUT = Conditional<
         CT::Exact<TO, Decay<TypeOf<FROM>>>,
         const TypeOf<TO>&,
         const TypeOf<TO>
      >;
      OUT value = static_cast<OUT>(DenseCast(Inner::GetFirst(from)));

   #if LANGULUS_SIMD(128BIT)
      if constexpr (CT::SIMD128i<REGISTER>) {
         if constexpr (CT::Integer8<TO>)
            return simde_mm_set1_epi8(value);
         else if constexpr (CT::UnsignedInteger8<TO>)
            return simde_x_mm_set1_epu8(value);
         else if constexpr (CT::Integer16<TO>)
            return simde_mm_set1_epi16(value);
         else if constexpr (CT::UnsignedInteger16<TO>)
            return simde_x_mm_set1_epu16(value);
         else if constexpr (CT::Integer32<TO>)
            return simde_mm_set1_epi32(value);
         else if constexpr (CT::UnsignedInteger32<TO>)
            return simde_x_mm_set1_epu32(value);
         else if constexpr (CT::Integer64<TO>)
            return simde_mm_set1_epi64x(value);
         else if constexpr (CT::UnsignedInteger64<TO>)
            return simde_x_mm_set1_epu64(value);
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m128i");
      }
      else if constexpr (CT::SIMD128f<REGISTER>) {
         if constexpr (CT::Float<TO>)
            return simde_mm_broadcast_ss(&value);
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m128");
      }
      else if constexpr (CT::SIMD128d<REGISTER>) {
         if constexpr (CT::Double<TO>)
            return simde_mm_set1_pd(value);
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m128d");
      }
      else
   #endif

   #if LANGULUS_SIMD(256BIT)
      if constexpr (CT::SIMD256i<REGISTER>) {
         if constexpr (CT::Integer8<TO>)
            return simde_mm256_set1_epi8(value);
         else if constexpr (CT::Integer16<TO>)
            return simde_mm256_set1_epi16(value);
         else if constexpr (CT::Integer32<TO>)
            return simde_mm256_set1_epi32(value);
         else if constexpr (CT::Integer64<TO>)
            return simde_mm256_set1_epi64x(value);
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m256i");
      }
      else if constexpr (CT::SIMD256f<REGISTER>) {
         if constexpr (CT::Float<TO>)
            return simde_mm256_broadcast_ss(&value);
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill __m256");
      }
      else if constexpr (CT::SIMD256d<REGISTER>) {
         if constexpr (CT::Double<TO>)
            return simde_mm256_broadcast_sd(&value);
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m256d");
      }
      else
   #endif

   #if LANGULUS_SIMD(512BIT)
      if constexpr (CT::SIMD512i<REGISTER>) {
         if constexpr (CT::Integer8<TO>)
            return simde_mm512_set1_epi8(value);
         else if constexpr (CT::Integer16<TO>)
            return simde_mm512_set1_epi16(value);
         else if constexpr (CT::Integer32<TO>)
            return simde_mm512_set1_epi32(value);
         else if constexpr (CT::Integer64<TO>)
            return simde_mm512_set1_epi64(value);
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m512i");
      }
      else if constexpr (CT::SIMD512f<REGISTER>) {
         if constexpr (CT::Float<TO>)
            return simde_mm512_set1_ps(value);
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill __m512");
      }
      else if constexpr (CT::SIMD512d<REGISTER>) {
         if constexpr (CT::Double<TO>)
            return simde_mm512_set1_pd(&value);
         else
            LANGULUS_ERROR("Unsupported type for SIMD::Fill of __m512d");
      }
      else
   #endif
      LANGULUS_ERROR("Bad REGISTER type for SIMD::Fill");
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
