///                                                                           
/// Langulus::SIMD                                                            
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>                    
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Intrinsics.hpp"
#include "IgnoreWarningsPush.inl"

namespace Langulus::SIMD
{

   /// Save a register to memory                                              
   ///   @tparam FROM - the register to save                                  
   ///   @tparam ALIGNED - whether or not 'to' array is aligned to Alignment  
   ///   @tparam T - the type of data to write (deducible)                    
   ///   @tparam S - the number of elements to write (deducible)              
   ///   @param from - the source register                                    
   ///   @param to - the destination array                                    
   template<CT::TSIMD FROM, bool ALIGNED = false, class T, Count S>
   LANGULUS(ALWAYSINLINE) void Store(const FROM& from, T(&to)[S]) noexcept {
      static_assert(S > 1, "Storing less than two elements is suboptimal "
         "- avoid SIMD operations on such arrays as a whole");
      constexpr Size toSize = sizeof(Decay<T>) * S;

   //                                                                   
   // __m128*                                                           
   //                                                                   
   #if LANGULUS_SIMD(128BIT)
      if constexpr (CT::Same<FROM, simde__m128>) {
         if constexpr (CT::Dense<T> && toSize == 16) {
            // Save to a dense array                                    
            if constexpr (ALIGNED)
               simde_mm_store_ps(to, from);
            else
               simde_mm_storeu_ps(to, from);
         }
         else {
            // Save to a sparse array, or a differently sized array     
            alignas(16) float temp[4];
            simde_mm_store_ps(temp, from);
            if constexpr (CT::Dense<T>)
               ::std::memcpy(to, temp, toSize);
            else {
               auto toIt = to;
               auto fromIt = temp;
               const auto toItEnd = to + S;
               while (toIt != toItEnd)
                  **(toIt++) = DenseCast(fromIt++);
            }
         }
      }
      else if constexpr (CT::Same<FROM, simde__m128d>) {
         if constexpr (CT::Dense<T> && toSize == 16) {
            // Save to a dense array                                    
            if constexpr (ALIGNED)
               simde_mm_store_pd(to, from);
            else
               simde_mm_storeu_pd(to, from);
         }
         else {
            // Save to a sparse array, or a differently sized array     
            simde_mm_storel_pd(SparseCast(to[0]), from);
            if constexpr (S > 1)
               simde_mm_storeh_pd(SparseCast(to[1]), from);
         }
      }
      else if constexpr (CT::Same<FROM, simde__m128i>) {
         if constexpr (CT::Dense<T> && toSize == 16) {
            // Save to a dense array                                    
            if constexpr (ALIGNED)
               simde_mm_store_si128(to, from);
            else
               simde_mm_storeu_si128(to, from);
         }
         else {
            // Save to a sparse array, or a differently sized array     
            alignas(16) Byte temp[16];
            simde_mm_store_si128(reinterpret_cast<simde__m128i*>(temp), from);
            if constexpr (CT::Dense<T>)
               ::std::memcpy(to, temp, toSize);
            else {
               auto toIt = to;
               auto fromIt = reinterpret_cast<Decay<T>*>(temp);
               const auto toItEnd = to + S;
               while (toIt != toItEnd)
                  **(toIt++) = DenseCast(fromIt++);
            }
         }
      }
      else
   #endif

   #if LANGULUS_SIMD(256BIT)
      //                                                                
      // __m256*                                                        
      //                                                                
      if constexpr (CT::Same<FROM, simde__m256>) {
         if constexpr (CT::Dense<T> && toSize == 32) {
            // Save to a dense array                                    
            if constexpr (ALIGNED)
               simde_mm256_store_ps(to, from);
            else
               simde_mm256_storeu_ps(to, from);
         }
         else {
            // Save to a sparse array, or a differently sized array     
            alignas(32) float temp[8];
            simde_mm256_store_ps(temp, from);
            if constexpr (CT::Dense<T>)
               ::std::memcpy(to, temp, toSize);
            else {
               auto toIt = to;
               auto fromIt = temp;
               const auto toItEnd = to + S;
               while (toIt != toItEnd)
                  **(toIt++) = DenseCast(fromIt++);
            }
         }
      }
      else if constexpr (CT::Same<FROM, simde__m256d>) {
         if constexpr (CT::Dense<T> && toSize == 32) {
            // Save to a dense array                                    
            if constexpr (ALIGNED)
               simde_mm256_store_pd(to, from);
            else
               simde_mm256_storeu_pd(to, from);
         }
         else {
            // Save to a sparse array, or a differently sized array     
            alignas(32) double temp[4];
            simde_mm256_store_pd(temp, from);
            if constexpr (CT::Dense<T>)
               ::std::memcpy(to, temp, toSize);
            else {
               auto toIt = to;
               auto fromIt = temp;
               const auto toItEnd = to + S;
               while (toIt != toItEnd)
                  **(toIt++) = DenseCast(fromIt++);
            }
         }
      }
      else if constexpr (CT::Same<FROM, simde__m256i>) {
         if constexpr (CT::Dense<T> && toSize == 32) {
            // Save to a dense array                                    
            if constexpr (ALIGNED)
               simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(to), from);
            else
               simde_mm256_storeu_si256(to, from);
         }
         else {
            // Save to a sparse array, or a differently sized array     
            alignas(32) Byte temp[32];
            simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(temp), from);
            if constexpr (CT::Dense<T>)
               ::std::memcpy(to, temp, toSize);
            else {
               auto toIt = to;
               auto fromIt = reinterpret_cast<Decay<T>*>(temp);
               const auto toItEnd = to + S;
               while (toIt != toItEnd)
                  **(toIt++) = DenseCast(fromIt++);
            }
         }
      }
      else
   #endif

   #if LANGULUS_SIMD(512BIT)
      //                                                                
      // __m512*                                                        
      //                                                                
      if constexpr (CT::Same<FROM, simde__m512>) {
         if constexpr (CT::Dense<T> && toSize == 64) {
            // Save to a dense array                                    
            if constexpr (ALIGNED)
               simde_mm512_store_ps(to, from);
            else
               simde_mm512_storeu_ps(to, from);
         }
         else {
            // Save to a sparse array, or a differently sized array     
            alignas(64) float temp[16];
            simde_mm512_store_ps(temp, from);
            if constexpr (CT::Dense<T>)
               ::std::memcpy(to, temp, toSize);
            else {
               auto toIt = to;
               auto fromIt = temp;
               const auto toItEnd = to + S;
               while (toIt != toItEnd)
                  **(toIt++) = DenseCast(fromIt++);
            }
         }
      }
      else if constexpr (CT::Same<FROM, simde__m512d>) {
         if constexpr (CT::Dense<T> && toSize == 64) {
            // Save to a dense array                                    
            if constexpr (ALIGNED)
               simde_mm512_store_pd(to, from);
            else
               simde_mm512_storeu_pd(to, from);
         }
         else {
            // Save to a sparse array, or a differently sized array     
            alignas(64) double temp[8];
            simde_mm512_store_pd(temp, from);
            if constexpr (CT::Dense<T>)
               ::std::memcpy(to, temp, toSize);
            else {
               auto toIt = to;
               auto fromIt = temp;
               const auto toItEnd = to + S;
               while (toIt != toItEnd)
                  **(toIt++) = DenseCast(fromIt++);
            }
         }
      }
      else if constexpr (CT::Same<FROM, simde__m512i>) {
         if constexpr (CT::Dense<T> && toSize == 64) {
            // Save to a dense array                                    
            if constexpr (ALIGNED)
               simde_mm512_store_si512(to, from);
            else
               simde_mm512_storeu_si512(to, from);
         }
         else {
            // Save to a sparse array, or a differently sized array     
            alignas(64) Byte temp[64];
            simde_mm512_store_si512(temp, from);
            if constexpr (CT::Dense<T>)
               ::std::memcpy(to, temp, toSize);
            else {
               auto toIt = to;
               auto fromIt = reinterpret_cast<Decay<T>*>(temp);
               const auto toItEnd = to + S;
               while (toIt != toItEnd)
                  **(toIt++) = DenseCast(fromIt++);
            }
         }
      }
      else
   #endif
         LANGULUS_ERROR("Unsupported FROM register for SIMD::Store");
   }

   
   /// Generalized store routine                                              
   ///   @tparam FROM - any source type, SIMD register, std::array or scalar  
   ///   @tparam TO - any destination type, array or scalar                   
   ///   @param from - what to store                                          
   ///   @param to - where to store it                                        
   template<class FROM, class TO>
   LANGULUS(ALWAYSINLINE) void GeneralStore(const FROM& from, TO& to) noexcept {
      if constexpr (CT::TSIMD<FROM>) {
         // Extract from SIMD register (produced from SIMD routine)     
         Store(from, to);
      }
      else if constexpr (!CT::Array<TO>) {
         // Extract from a single number (produced from fallback)       
         // Or, alternatively, when doing compare SIMDs                 
         DenseCast(to) = from;
      }
      else if constexpr (ExtentOf<TO> == 1) {
         // Extract from a scalar (produced from fallback routine)      
         // But write results to an output array of size 1              
         DenseCast(to[0]) = from;
      }
      else if constexpr (CT::Sparse<Deext<TO>>) {
         // Extract from std::array (produced from fallback routine)    
         // But write results in a sparse output array                  
         for (int i = 0; i < ExtentOf<TO>; ++i)
            *to[i] = from[i];
      }
      else {
         // Extract from std::array (produced from fallback routine)    
         // Write results to a dense output array (fastest)             
         ::std::memcpy(to, from.data(), sizeof(to));
      }
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"