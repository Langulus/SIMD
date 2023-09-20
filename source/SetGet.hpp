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
   namespace Inner
   {

      /// Get an element of an array, or zero if out of range                 
      ///   @tparam DEF - the default value for the element, if out of S      
      ///   @tparam IDX - the index to get                                    
      ///   @tparam MAXS - the maximum number of elements T inside register   
      ///   @tparam REVERSE - whether or not to count in inverse              
      ///   @tparam T - the type of an array element                          
      ///   @tparam S - the size of the array                                 
      ///   @param values - the array to access                               
      ///   @return a reference to the element, or DEF if out of range        
      template<class R, int DEF, Offset IDX, Count MAXS, bool REVERSE = false, class T, Count S>
      LANGULUS(INLINED)
      const R& Get(const T(&values)[S]) {
         static_assert(S <= MAXS, "S must be in MAXS limit");
         static_assert(IDX < MAXS, "IDX must be in MAXS limit");
         static constinit auto fallback = static_cast<R>(DEF);

         if constexpr (REVERSE) {
            if constexpr (MAXS - IDX - 1 < S)
               return reinterpret_cast<const R&>(DenseCast(values[MAXS - IDX - 1]));
            else
               return fallback;
         }
         else {
            if constexpr (IDX < S)
               return reinterpret_cast<const R&>(DenseCast(values[IDX]));
            else
               return fallback;
         }
      }

      /// Inner array iteration and register setting                          
      ///   @tparam DEF - the default value for the element, if out of S      
      ///   @tparam CHUNK - the size of register to fill (in bytes)           
      ///   @tparam T - the type of an array element                          
      ///   @tparam S - the size of the array                                 
      ///   @tparam INDICES - the indices to use                              
      ///   @param values - the array to access                               
      ///   @return the register                                              
      template<int DEF, Size CHUNK, class T, Count S, Offset... INDICES>
      LANGULUS(INLINED)
      auto Set(std::integer_sequence<Offset, INDICES...>, const T(&values)[S]) {
         #if LANGULUS_SIMD(128BIT)
            if constexpr (CHUNK == 16) {
               LANGULUS_SIMD_VERBOSE("Setting 128bit register from ", S, " elements");

               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm_setr_epi8(Get<int8_t, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::UnsignedInteger8<T>)
                  return simde_mm_setr_epi8(Get<int8_t, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::SignedInteger16<T>)
                  return simde_mm_setr_epi16(Get<int16_t, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::UnsignedInteger16<T>)
                  return simde_mm_setr_epi16(Get<int16_t, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::SignedInteger32<T>)
                  return simde_mm_setr_epi32(Get<int32_t, DEF, INDICES, 4>(values)...);
               else if constexpr (CT::UnsignedInteger32<T>)
                  return simde_mm_setr_epi32(Get<int32_t, DEF, INDICES, 4>(values)...);
               else if constexpr (CT::SignedInteger64<T>)
                  return simde_mm_set_epi64x(Get<int64_t, DEF, INDICES, 2, true>(values)...);
               else if constexpr (CT::UnsignedInteger64<T>)
                  return simde_mm_set_epi64x(Get<int64_t, DEF, INDICES, 2, true>(values)...);
               else if constexpr (CT::Float<T>)
                  return simde_mm_setr_ps(Get<simde_float32, DEF, INDICES, 4>(values)...);
               else if constexpr (CT::Double<T>)
                  return simde_mm_setr_pd(Get<simde_float64, DEF, INDICES, 2>(values)...);
               else
                  LANGULUS_ERROR("Can't set 16-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(256BIT)
            if constexpr (CHUNK == 32) {
               LANGULUS_SIMD_VERBOSE("Setting 256bit register from ", S, " elements");

               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm256_setr_epi8(Get<int8_t, DEF, INDICES, 32>(values)...);
               else if constexpr (CT::UnsignedInteger8<T>)
                  return simde_mm256_setr_epi8(Get<int8_t, DEF, INDICES, 32>(values)...);
               else if constexpr (CT::SignedInteger16<T>)
                  return simde_mm256_setr_epi16(Get<int16_t, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::UnsignedInteger16<T>)
                  return simde_mm256_setr_epi16(Get<int16_t, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::SignedInteger32<T>)
                  return simde_mm256_setr_epi32(Get<int32_t, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::UnsignedInteger32<T>)
                  return simde_mm256_setr_epi32(Get<int32_t, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::SignedInteger64<T>)
                  return simde_mm256_setr_epi64x(Get<int64_t, DEF, INDICES, 4>(values)...);
               else if constexpr (CT::UnsignedInteger64<T>)
                  return simde_mm256_setr_epi64x(Get<int64_t, DEF, INDICES, 4>(values)...);
               else if constexpr (CT::Float<T>)
                  return simde_mm256_setr_ps(Get<simde_float32, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::Double<T>)
                  return simde_mm256_setr_pd(Get<simde_float64, DEF, INDICES, 4>(values)...);
               else
                  LANGULUS_ERROR("Can't set 32-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(512BIT)
            if constexpr (CHUNK == 64) {
               LANGULUS_SIMD_VERBOSE("Setting 512bit register from ", S, " elements");

               if constexpr (CT::SignedInteger8<T>)
                  return simde_mm512_setr_epi8(Get<int8_t, DEF, INDICES, 64>(values)...);
               else if constexpr (CT::UnsignedInteger8<T>)
                  return simde_mm512_setr_epi8(Get<int8_t, DEF, INDICES, 64>(values)...);
               else if constexpr (CT::SignedInteger16<T>)
                  return simde_mm512_setr_epi16(Get<int16_t, DEF, INDICES, 32>(values)...);
               else if constexpr (CT::UnsignedInteger16<T>)
                  return simde_mm512_setr_epi16(Get<int16_t, DEF, INDICES, 32>(values)...);
               else if constexpr (CT::SignedInteger32<T>)
                  return simde_mm512_setr_epi32(Get<int32_t, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::UnsignedInteger32<T>)
                  return simde_mm512_setr_epi32(Get<int32_t, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::SignedInteger64<T>)
                  return simde_mm512_setr_epi64(Get<int64_t, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::UnsignedInteger64<T>)
                  return simde_mm512_setr_epi64(Get<int64_t, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::Float<T>)
                  return simde_mm512_setr_ps(Get<simde_float32, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::Double<T>)
                  return simde_mm512_setr_pd(Get<simde_float64, DEF, INDICES, 8>(values)...);
               else
                  LANGULUS_ERROR("Can't set 64-byte package");
            }
            else
         #endif
            LANGULUS_ERROR("Unsupported package");
      }

   } // namespace Langulus::SIMD::Inner


   /// Construct a register manually                                          
   ///   @tparam CHUNK - the size of the chunk to set                         
   ///   @tparam T - the type of the array element                            
   ///   @tparam S - the size of the array                                    
   ///   @param values - the array to wrap                                    
   ///   @return the register                                                 
   template<int DEF, Size CHUNK, class T, Count S>
   LANGULUS(INLINED)
   auto Set(const T(&values)[S]) noexcept {
      if constexpr (S < 2) {
         // No point in storing a single value in a large register      
         // If this is reached, then the library did not choose the     
         // optimal route for your operation at compile time            
         return Unsupported {};
      }
      else {
         constexpr auto MaxS = CHUNK / sizeof(Decay<T>);
         static_assert((CT::Dense<T> and MaxS > S) or (CT::Sparse<T> and MaxS >= S),
            "S should be smaller (or equal if sparse) than MaxS - use load otherwise");
         return Inner::Set<DEF, CHUNK>(::std::make_integer_sequence<Count, MaxS> {}, values);
      }
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
