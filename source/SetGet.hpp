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
      ///   @tparam FROM - the scalar/array/vector to use for setting         
      ///   @param values - the array to access                               
      ///   @return a reference to the element, or DEF if out of range        
      template<class R, int DEF, Offset IDX, Count MAXS, bool REVERSE = false, class FROM>
      LANGULUS(INLINED)
      constexpr decltype(auto) Get(const FROM& values) {
         constexpr auto S = CountOf<FROM>;
         static_assert( S <= MAXS, "S must be in MAXS limit");
         static_assert(IDX < MAXS, "IDX must be in MAXS limit");

         if constexpr (REVERSE) {
            if constexpr (MAXS - IDX - 1 < S) {
               LANGULUS_SIMD_VERBOSE("Setting [", IDX, "] to ", DenseCast(values[MAXS - IDX - 1]));
               return reinterpret_cast<const R&>(
                  DenseCast(values[MAXS - IDX - 1]));
            }
            else {
               LANGULUS_SIMD_VERBOSE("Setting [", IDX, "] to ", static_cast<R>(DEF));
               return static_cast<R>(DEF);
            }
         }
         else {
            if constexpr (IDX < S) {
               LANGULUS_SIMD_VERBOSE("Setting [", IDX, "] to ", DenseCast(values[IDX]));
               return reinterpret_cast<const R&>(
                  DenseCast(values[IDX]));
            }
            else {
               LANGULUS_SIMD_VERBOSE("Setting [", IDX, "] to ", static_cast<R>(DEF));
               return static_cast<R>(DEF);
            }
         }
      }

      /// Inner array iteration and register setting                          
      ///   @tparam DEF - the default value for the element, if out of S      
      ///   @tparam CHUNK - the size of register to fill (in bytes)           
      ///   @tparam FROM - the scalar/array/vector to use for setting         
      ///   @tparam INDICES - the indices to use                              
      ///   @param values - the array to access                               
      ///   @return the register                                              
      template<int DEF, Size CHUNK, CT::Vector FROM, Offset... INDICES>
      LANGULUS(INLINED)
      auto Set(std::integer_sequence<Offset, INDICES...>, const FROM& values) {
         using T = Decay<TypeOf<FROM>>;

         #if LANGULUS_SIMD(128BIT)
            if constexpr (CHUNK == 16) {
               LANGULUS_SIMD_VERBOSE("Setting 128bit register from ", CountOf<FROM>, " elements");

               if constexpr (CT::Integer8<T>)
                  return simde_mm_setr_epi8(
                     Get<int8_t, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::Integer16<T>)
                  return simde_mm_setr_epi16(
                     Get<int16_t, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm_setr_epi32(
                     Get<int32_t, DEF, INDICES, 4>(values)...);
               else if constexpr (CT::Integer64<T>)
                  return simde_mm_set_epi64x(
                     Get<int64_t, DEF, INDICES, 2, true>(values)...);
               else if constexpr (CT::Float<T>)
                  return simde_mm_setr_ps(
                     Get<simde_float32, DEF, INDICES, 4>(values)...);
               else if constexpr (CT::Double<T>)
                  return simde_mm_setr_pd(
                     Get<simde_float64, DEF, INDICES, 2>(values)...);
               else LANGULUS_ERROR("Can't set 16-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(256BIT)
            if constexpr (CHUNK == 32) {
               LANGULUS_SIMD_VERBOSE("Setting 256bit register from ", CountOf<FROM>, " elements");

               if constexpr (CT::Integer8<T>)
                  return simde_mm256_setr_epi8(
                     Get<int8_t, DEF, INDICES, 32>(values)...);
               else if constexpr (CT::Integer16<T>)
                  return simde_mm256_setr_epi16(
                     Get<int16_t, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm256_setr_epi32(
                     Get<int32_t, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::Integer64<T>)
                  return simde_mm256_setr_epi64x(
                     Get<int64_t, DEF, INDICES, 4>(values)...);
               else if constexpr (CT::Float<T>)
                  return simde_mm256_setr_ps(
                     Get<simde_float32, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::Double<T>)
                  return simde_mm256_setr_pd(
                     Get<simde_float64, DEF, INDICES, 4>(values)...);
               else LANGULUS_ERROR("Can't set 32-byte package");
            }
            else
         #endif

         #if LANGULUS_SIMD(512BIT)
            if constexpr (CHUNK == 64) {
               LANGULUS_SIMD_VERBOSE("Setting 512bit register from ", CountOf<FROM>, " elements");

               if constexpr (CT::Integer8<T>)
                  return simde_mm512_setr_epi8(
                     Get<int8_t, DEF, INDICES, 64>(values)...);
               else if constexpr (CT::Integer16<T>)
                  return simde_mm512_setr_epi16(
                     Get<int16_t, DEF, INDICES, 32>(values)...);
               else if constexpr (CT::Integer32<T>)
                  return simde_mm512_setr_epi32(
                     Get<int32_t, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::Integer64<T>)
                  return simde_mm512_setr_epi64(
                     Get<int64_t, DEF, INDICES, 8>(values)...);
               else if constexpr (CT::Float<T>)
                  return simde_mm512_setr_ps(
                     Get<simde_float32, DEF, INDICES, 16>(values)...);
               else if constexpr (CT::Double<T>)
                  return simde_mm512_setr_pd(
                     Get<simde_float64, DEF, INDICES, 8>(values)...);
               else LANGULUS_ERROR("Can't set 64-byte package");
            }
            else
         #endif
            LANGULUS_ERROR("Unsupported package");
      }

   } // namespace Langulus::SIMD::Inner


   /// Construct a register manually                                          
   ///   @tparam DEF - the default value for the element, if out of S         
   ///   @tparam CHUNK - the size of the chunk to set                         
   ///   @tparam FROM - the scalar/array/vector to use for setting            
   ///   @param values - the array to wrap                                    
   ///   @return the register                                                 
   template<int DEF = 0, Size CHUNK = Alignment, CT::Vector FROM>
   LANGULUS(INLINED)
   auto Set(const FROM& values) noexcept {
      using T = TypeOf<FROM>;
      constexpr auto S = CountOf<FROM>;
      constexpr auto MaxS = CHUNK / sizeof(Decay<T>);
      static_assert((CT::Dense<T>  and MaxS >  S)
                 or (CT::Sparse<T> and MaxS >= S),
         "S should be smaller (or equal if sparse) than MaxS - use load otherwise");

      return Inner::Set<DEF, CHUNK>(
         ::std::make_integer_sequence<Count, MaxS> {},
         values
      );
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
