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
#include <ranges>


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Save a register to memory                                           
      ///   @tparam FROM - the register to save                               
      ///   @tparam TO - the type of data to write (deducible)                
      ///   @param from - the source register                                 
      ///   @param to - the destination array                                 
      template<CT::SIMD FROM, CT::NotSIMD TO>
      LANGULUS(INLINED)
      void Store(UNUSED() const FROM& from, UNUSED() TO& to) noexcept {
         constexpr auto S = CountOf<TO>;
         static_assert(S > 1, 
            "Storing less than two elements is suboptimal "
            "- avoid SIMD operations on such arrays as a whole"
         );
         using T = TypeOf<TO>;
         using DT = Decay<T>;
         UNUSED() constexpr Size denseSize = sizeof(DT) * S;

      #if LANGULUS_SIMD(128BIT)
         //                                                             
         // __m128*                                                     
         //                                                             
         if constexpr (CT::SIMD128f<FROM>) {
            if constexpr (CT::Dense<T> and denseSize == 16) {
               // Save to a dense array                                 
               if constexpr (alignof(TO) % 16 == 0) {
                  LANGULUS_SIMD_VERBOSE("Storing full 128f to aligned ", denseSize, " bytes");
                  auto to_ps = reinterpret_cast<simde_float32*>(&Inner::GetFirst(to));
                  simde_mm_store_ps(to_ps, from);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing full 128f to unaligned ", denseSize, " bytes");
                  simde_mm_storeu_ps(&Inner::GetFirst(to), from);
               }
            }
            else if constexpr (denseSize <= 16) {
               // Save to a sparse array, or a differently sized array  
               alignas(16) simde_float32 temp[4];
               simde_mm_store_ps(temp, from);

               if constexpr (CT::Dense<T>) {
                  LANGULUS_SIMD_VERBOSE("Storing partial 128f to ", denseSize, " bytes");
                  ::std::memcpy(&Inner::GetFirst(to), temp, denseSize);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing partial 128f to sparse array of size ", S);
                  auto toIt = to;
                  auto fromIt = temp;
                  const auto toItEnd = to + S;
                  while (toIt != toItEnd)
                     **(toIt++) = DenseCast(fromIt++);
               }
            }
            else LANGULUS_ERROR("Output size too big");
         }
         else if constexpr (CT::SIMD128d<FROM>) {
            if constexpr (CT::Dense<T> and denseSize == 16) {
               // Save to a dense array                                 
               if constexpr (alignof(TO) % 16 == 0) {
                  LANGULUS_SIMD_VERBOSE("Storing full 128d to aligned ", denseSize, " bytes");
                  auto to_pd = reinterpret_cast<simde_float64*>(&Inner::GetFirst(to));
                  simde_mm_store_pd(to_pd, from);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing full 128d to unaligned ", denseSize, " bytes");
                  simde_mm_storeu_pd(&Inner::GetFirst(to), from);
               }
            }
            else if constexpr (denseSize <= 16) {
               if constexpr (CT::Dense<T>)
                  LANGULUS_SIMD_VERBOSE("Storing partial 128d to ", denseSize, " bytes");
               else
                  LANGULUS_SIMD_VERBOSE("Storing partial 128d to sparse array of size ", S);

               // Save to a sparse array, or a differently sized array  
               simde_mm_storel_pd(SparseCast(to[0]), from);
               if constexpr (S > 1)
                  simde_mm_storeh_pd(SparseCast(to[1]), from);
            }
            else LANGULUS_ERROR("Output size too big");
         }
         else if constexpr (CT::SIMD128i<FROM>) {
            if constexpr (CT::Dense<T> and denseSize == 16) {
               // Save to a dense array                                 
               if constexpr (alignof(TO) % 16 == 0) {
                  LANGULUS_SIMD_VERBOSE("Storing full 128i to aligned ", denseSize, " bytes");
                  auto to_si = reinterpret_cast<simde__m128i*>(&Inner::GetFirst(to));
                  simde_mm_store_si128(to_si, from);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing full 128i to unaligned ", denseSize, " bytes");
                  simde_mm_storeu_si128(&Inner::GetFirst(to), from);
               }
            }
            else if constexpr (denseSize <= 16) {
               // Save to a sparse array, or a differently sized array  
               alignas(16) DT temp[16 / sizeof(DT)];
               simde_mm_store_si128(reinterpret_cast<simde__m128i*>(temp), from);

               if constexpr (CT::Dense<T>) {
                  LANGULUS_SIMD_VERBOSE("Storing partial 128i to ", denseSize, " bytes");
                  ::std::memcpy(&Inner::GetFirst(to), temp, denseSize);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing partial 128d to sparse array of size ", S);
                  auto toIt = to;
                  auto fromIt = temp;
                  const auto toItEnd = to + S;
                  while (toIt != toItEnd)
                     **(toIt++) = *(fromIt++);
               }
            }
            else LANGULUS_ERROR("Output size too big");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         //                                                             
         // __m256*                                                     
         //                                                             
         if constexpr (CT::SIMD256f<FROM>) {
            if constexpr (CT::Dense<T> and denseSize == 32) {
               // Save to a dense array                                 
               if constexpr (alignof(TO) % 32 == 0) {
                  LANGULUS_SIMD_VERBOSE("Storing full 256f to aligned ", denseSize, " bytes");
                  auto to_ps = reinterpret_cast<simde_float32*>(&Inner::GetFirst(to));
                  simde_mm256_store_ps(to_ps, from);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing full 256f to unaligned ", denseSize, " bytes");
                  simde_mm256_storeu_ps(&Inner::GetFirst(to), from);
               }
            }
            else if constexpr (denseSize <= 32) {
               // Save to a sparse array, or a differently sized array  
               alignas(32) simde_float32 temp[8];
               simde_mm256_store_ps(temp, from);

               if constexpr (CT::Dense<T>) {
                  LANGULUS_SIMD_VERBOSE("Storing partial 256f to ", denseSize, " bytes");
                  ::std::memcpy(&Inner::GetFirst(to), temp, denseSize);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing partial 256f to sparse array of size ", S);
                  auto toIt = to;
                  auto fromIt = temp;
                  const auto toItEnd = to + S;
                  while (toIt != toItEnd)
                     **(toIt++) = DenseCast(fromIt++);
               }
            }
            else LANGULUS_ERROR("Output size too big");
         }
         else if constexpr (CT::SIMD256d<FROM>) {
            if constexpr (CT::Dense<T> and denseSize == 32) {
               // Save to a dense array                                 
               if constexpr (alignof(TO) % 32 == 0) {
                  LANGULUS_SIMD_VERBOSE("Storing full 256d to aligned ", denseSize, " bytes");
                  auto to_pd = reinterpret_cast<simde_float64*>(&Inner::GetFirst(to));
                  simde_mm256_store_pd(to_pd, from);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing full 256d to unaligned ", denseSize, " bytes");
                  simde_mm256_storeu_pd(&Inner::GetFirst(to), from);
               }
            }
            else if constexpr (denseSize <= 32) {
               // Save to a sparse array, or a differently sized array  
               alignas(32) simde_float64 temp[4];
               simde_mm256_store_pd(temp, from);

               if constexpr (CT::Dense<T>) {
                  LANGULUS_SIMD_VERBOSE("Storing partial 256d to ", denseSize, " bytes");
                  ::std::memcpy(&Inner::GetFirst(to), temp, denseSize);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing partial 256d to sparse array of size ", S);
                  auto toIt = to;
                  auto fromIt = temp;
                  const auto toItEnd = to + S;
                  while (toIt != toItEnd)
                     **(toIt++) = DenseCast(fromIt++);
               }
            }
            else LANGULUS_ERROR("Output size too big");
         }
         else if constexpr (CT::SIMD256i<FROM>) {
            if constexpr (CT::Dense<T> and denseSize == 32) {
               // Save to a dense array                                 
               if constexpr (alignof(TO) % 32 == 0) {
                  LANGULUS_SIMD_VERBOSE("Storing full 256i to aligned ", denseSize, " bytes");
                  auto to_si = reinterpret_cast<simde__m256i*>(&Inner::GetFirst(to));
                  simde_mm256_store_si256(to_si, from);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing full 256i to unaligned ", denseSize, " bytes");
                  simde_mm256_storeu_si256(&Inner::GetFirst(to), from);
               }
            }
            else if constexpr (denseSize <= 32) {
               // Save to a sparse array, or a differently sized array  
               alignas(32) DT temp[32 / sizeof(DT)];
               simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(temp), from);

               if constexpr (CT::Dense<T>) {
                  LANGULUS_SIMD_VERBOSE("Storing partial 256i to ", denseSize, " bytes; "
                     "from ", NameOf<FROM>(), " of size ", sizeof(FROM), " to ", NameOf<TO>(),
                     " of size ", sizeof(TO), " (aka ", NameOf<DT>(), "[", S, "] of size ", denseSize,")"
                  );

                  ::std::memcpy(&Inner::GetFirst(to), temp, denseSize);
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing partial 256i to sparse array of size ", S);
                  auto toIt = to;
                  auto fromIt = temp;
                  const auto toItEnd = to + S;
                  while (toIt != toItEnd)
                     **(toIt++) = *(fromIt++);
               }
            }
            else LANGULUS_ERROR("Output size too big");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         //                                                             
         // __m512*                                                     
         //                                                             
         if constexpr (CT::SIMD512f<FROM>) {
            if constexpr (CT::Dense<T> and denseSize == 64) {
               // Save to a dense array                                 
               if constexpr (alignof(TO) % 64 == 0) {
                  auto to_ps = reinterpret_cast<simde_float32*>(&Inner::GetFirst(to));
                  simde_mm512_store_ps(to_ps, from);
               }
               else simde_mm512_storeu_ps(&Inner::GetFirst(to), from);
            }
            else if constexpr (denseSize <= 64) {
               // Save to a sparse array, or a differently sized array  
               alignas(64) simde_float32 temp[16];
               simde_mm512_store_ps(temp, from);

               if constexpr (CT::Dense<T>)
                  ::std::memcpy(&Inner::GetFirst(to), temp, denseSize);
               else {
                  auto toIt = to;
                  auto fromIt = temp;
                  const auto toItEnd = to + S;
                  while (toIt != toItEnd)
                     **(toIt++) = DenseCast(fromIt++);
               }
            }
            else LANGULUS_ERROR("Output size too big");
         }
         else if constexpr (CT::SIMD512d<FROM>) {
            if constexpr (CT::Dense<T> and denseSize == 64) {
               // Save to a dense array                                 
               if constexpr (alignof(TO) % 64 == 0) {
                  auto to_pd = reinterpret_cast<simde_float64*>(&Inner::GetFirst(to));
                  simde_mm512_store_pd(to_pd, from);
               }
               else simde_mm512_storeu_pd(&Inner::GetFirst(to), from);
            }
            else if constexpr (denseSize <= 64) {
               // Save to a sparse array, or a differently sized array  
               alignas(64) simde_float64 temp[8];
               simde_mm512_store_pd(temp, from);

               if constexpr (CT::Dense<T>)
                  ::std::memcpy(&Inner::GetFirst(to), temp, denseSize);
               else {
                  auto toIt = to;
                  auto fromIt = temp;
                  const auto toItEnd = to + S;
                  while (toIt != toItEnd)
                     **(toIt++) = DenseCast(fromIt++);
               }
            }
            else LANGULUS_ERROR("Output size too big");
         }
         else if constexpr (CT::SIMD512i<FROM>) {
            if constexpr (CT::Dense<T> and denseSize == 64) {
               // Save to a dense array                                 
               if constexpr (alignof(TO) % 64 == 0) {
                  auto to_si = reinterpret_cast<simde__m512i*>(&Inner::GetFirst(to));
                  simde_mm512_store_si512(to_si, from);
               }
               else simde_mm512_storeu_si512(&Inner::GetFirst(to), from);
            }
            else if constexpr (denseSize <= 64) {
               // Save to a sparse array, or a differently sized array  
               alignas(64) DT temp[64 / sizeof(DT)];
               simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(temp), from);

               if constexpr (CT::Dense<T>)
                  ::std::memcpy(&Inner::GetFirst(to), temp, denseSize);
               else {
                  auto toIt = to;
                  auto fromIt = temp;
                  const auto toItEnd = to + S;
                  while (toIt != toItEnd)
                     **(toIt++) = *(fromIt++);
               }
            }
            else LANGULUS_ERROR("Output size too big");
         }
         else
      #endif
         LANGULUS_ERROR("Unsupported FROM register for SIMD::Store");
      }
   
   } // namespace Langulus::SIMD::Inner


   /// Fallback store routine, doesn't use SIMD                               
   ///   @tparam FROM - any source type, SIMD register, std::array,           
   ///                  boolvector, or scalar                                 
   ///   @tparam TO - any destination type, array or scalar                   
   ///   @param from - what to store                                          
   ///   @param to - where to store it                                        
   template<CT::NotSIMD FROM, CT::NotSIMD TO>
   LANGULUS(INLINED)
   constexpr void StoreConstexpr(const FROM& from, TO& to) noexcept {
      if constexpr (CT::Bitmask<FROM>) {
         // Extract from bitmask (produced from SIMD compare routine)   
         if constexpr (CT::Bitmask<TO>) {
            // Store in another bitmask                                 
            DenseCast(to) = from;
         }
         else if constexpr (CT::Bool<TO>) {
            // Convert each bit to a boolean inside an array            
            static_assert(CountOf<FROM> == CountOf<TO>, "Counts must match");
            for (Offset i = 0; i < CountOf<TO>; ++i)
               DenseCast(to[i]) = from[i];
         }
         else LANGULUS_ERROR("Bad output to store a bitmask");
      }
      else if constexpr (::std::ranges::range<FROM>) {
         // Extract from anything that has begin() and end() methods    
         if constexpr (CT::Bitmask<TO>) {
            // Store as bitmask                                         
            using T = typename Decay<TO>::Type;
            for (decltype(from.size()) i = 0; i < from.size(); ++i)
               DenseCast(to) |= (static_cast<T>(from[i]) << static_cast<T>(i));
         }
         else if constexpr (CountOf<TO> == 1) {
            // Store as a single number (produced from fallback)        
            if constexpr (CT::Bool<TO>) {
               // Short-circuit on first false flag (AND logic)         
               for (auto& it : from) {
                  if (not it) {
                     to = false;
                     return;
                  }
               }

               to = true;
            }
            else {
               // Otherwise just copy first element                     
               DenseCast(GetFirst(to)) = from[0];
            }
         }
         else {
            // Store to a sparse/dense output array                     
            static_assert(CountOf<FROM> == CountOf<TO>, "Counts must match");
            for (Count i = 0; i < CountOf<TO>; ++i)
               DenseCast(to[i]) = from[i];
         }
      }
      else {
         // Extract from a scalar                                       
         if constexpr (CountOf<TO> == 1) {
            if constexpr (CT::Bitmask<TO>) {
               // Store to a single-bit bitmask                         
               DenseCast(to) = from;
            }
            else {
               // Store to an output array of size 1, or scalar         
               DenseCast(Inner::GetFirst(to)) = from;
            }
         }
         else {
            // Multicast only result to an output array                 
            for (Count i = 0; i < CountOf<TO>; ++i)
               DenseCast(to[i]) = from;
         }
      }
   }

   /// Generalized store routine                                              
   /// Can be either constexpr or not, using SIMD or not                      
   ///   @tparam FROM - any source type, SIMD register, std::array,           
   ///                  boolvector, or scalar                                 
   ///   @tparam TO - any destination type, array or scalar                   
   ///   @param from - what to store                                          
   ///   @param to - where to store it                                        
   template<class FROM, CT::NotSIMD TO>
   LANGULUS(INLINED)
   constexpr void Store(const FROM& from, TO& to) noexcept {
      if constexpr (CT::SIMD<FROM>)
         Inner::Store(from, to);
      else
         StoreConstexpr(from, to);
   }

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"