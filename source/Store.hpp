///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "Common.hpp"
#include "Bitmask.hpp"


namespace Langulus::SIMD
{
   namespace Inner
   {

      /// Save a register to a bitmask in memory                              
      ///   @param from - the source register                                 
      ///   @param to - the destination vector                                
      LANGULUS(INLINED)
      void StoreSIMD(const CT::SIMD auto& from, CT::Bitmask auto& to) noexcept {
         using R  = Deref<decltype(from)>;
         using T  = TypeOf<R>;

         if constexpr (CT::SIMD128<R>) {
            if      constexpr (CT::Integer8<T>)    to = simde_mm_movemask_epi8   (from);
            else if constexpr (CT::Integer16<T>)   to = simde_mm_movemask_epi8   (simde_mm_packs_epi16(from, from.Zero()));
            else if constexpr (CT::Integer32<T>)   to = simde_mm_movemask_ps     (simde_mm_castsi128_ps(from));
            else if constexpr (CT::Integer64<T>)   to = simde_mm_movemask_pd     (simde_mm_castsi128_pd(from));
            else if constexpr (CT::Float<T>)       to = simde_mm_movemask_ps     (from);
            else if constexpr (CT::Double<T>)      to = simde_mm_movemask_pd     (from);
            else LANGULUS_ERROR("Unsupported type");
         }
         else if constexpr (CT::SIMD256<R>) {
            if      constexpr (CT::Integer8<T>)    to = simde_mm256_movemask_epi8(from);
            else if constexpr (CT::Integer16<T>)   to = simde_mm256_movemask_epi8(simde_mm256_packs_epi16(from, from.Zero()));
            else if constexpr (CT::Integer32<T>)   to = simde_mm256_movemask_ps  (simde_mm256_castsi256_ps(from));
            else if constexpr (CT::Integer64<T>)   to = simde_mm256_movemask_pd  (simde_mm256_castsi256_pd(from));
            else if constexpr (CT::Float<T>)       to = simde_mm256_movemask_ps  (from);
            else if constexpr (CT::Double<T>)      to = simde_mm256_movemask_pd  (from);
            else LANGULUS_ERROR("Unsupported type");
         }
         else if constexpr (CT::SIMD512<R>) {
            if      constexpr (CT::Integer8<T>)    to = simde_mm512_movemask_epi8(from);
            else if constexpr (CT::Integer16<T>)   to = simde_mm512_movemask_epi8(simde_mm512_packs_epi16(from, from.Zero()));
            else if constexpr (CT::Integer32<T>)   to = simde_mm512_movemask_ps  (simde_mm512_castsi256_ps(from));
            else if constexpr (CT::Integer64<T>)   to = simde_mm512_movemask_pd  (simde_mm512_castsi256_pd(from));
            else if constexpr (CT::Float<T>)       to = simde_mm512_movemask_ps  (from);
            else if constexpr (CT::Double<T>)      to = simde_mm512_movemask_pd  (from);
            else LANGULUS_ERROR("Unsupported type");
         }
         else LANGULUS_ERROR("Unsupported register");
      }

      /// Save a register to a vector in memory                               
      ///   @param from - the source register                                 
      ///   @param to - the destination vector                                
      LANGULUS(INLINED)
      void StoreSIMD(const CT::SIMD auto& from, CT::Vector auto& to) noexcept {
         using TO   = Deref<decltype(to)>;
         using TO_T = TypeOf<TO>;
         using R    = Deref<decltype(from)>;
         using T    = TypeOf<R>;

         static_assert(CountOf<TO> <= CountOf<R>,
            "Destination array must be smaller or equal of the register size");
         static_assert(CountOf<TO> > 1,
            "Storing a single element is suboptimial - don't use SIMD in the first place");
         static_assert(CT::Similar<T, TO_T> or CT::Bool<TO_T>,
            "Storing doesn't parform conversion, so destination must be "
            "of similar type as the register");

         if constexpr (CT::Bool<TO_T>) {
            // register -> bool array                                   
            Bitmask<CountOf<TO>> mask;
            StoreSIMD(from, mask);
            mask.AsVector(to);
         }
         else if constexpr (CT::SIMD128<R>) {
            if constexpr (CT::Float<T>) {
               // To float array                                        
               if constexpr (sizeof(to) == sizeof(from)) {
                  if constexpr (alignof(TO) % 16 == 0) {
                     LANGULUS_SIMD_VERBOSE("Storing 128f to aligned");
                     simde_mm_store_ps(&GetFirst(to), from);
                  }
                  else {
                     LANGULUS_SIMD_VERBOSE("Storing 128f to unaligned");
                     simde_mm_storeu_ps(&GetFirst(to), from);
                  }
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing 128f to partial");
                  alignas(16) T temp[CountOf<R>];
                  simde_mm_store_ps(temp, from);
                  memcpy(&GetFirst(to), temp, sizeof(to));
               }
            }
            else if constexpr (CT::Double<T>) {
               // To double array                                       
               if constexpr (sizeof(to) == sizeof(from)) {
                  if constexpr (alignof(TO) % 16 == 0) {
                     LANGULUS_SIMD_VERBOSE("Storing 128d to aligned");
                     simde_mm_store_pd(&GetFirst(to), from);
                  }
                  else {
                     LANGULUS_SIMD_VERBOSE("Storing 128d to unaligned");
                     simde_mm_storeu_pd(&GetFirst(to), from);
                  }
               }
               else LANGULUS_ERROR("Shouldn't be reached (storing one double)");
            }
            else if constexpr (CT::Integer<T>) {
               // To integer array                                      
               if constexpr (sizeof(to) == sizeof(from)) {
                  if constexpr (alignof(TO) % 16 == 0) {
                     LANGULUS_SIMD_VERBOSE("Storing 128i to aligned");
                     simde_mm_store_si128(reinterpret_cast<simde__m128i*>(&GetFirst(to)), from);
                  }
                  else {
                     LANGULUS_SIMD_VERBOSE("Storing 128i to unaligned");
                     simde_mm_storeu_si128(&GetFirst(to), from);
                  }
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing 128i to partial");
                  alignas(16) T temp[CountOf<R>];
                  simde_mm_store_si128(reinterpret_cast<simde__m128i*>(temp), from);
                  memcpy(&GetFirst(to), temp, sizeof(to));
               }
            }
            else LANGULUS_ERROR("Unsupported output");
         }
         else if constexpr (CT::SIMD256<R>) {
            if constexpr (CT::Float<T>) {
               // To float array                                        
               if constexpr (sizeof(to) == sizeof(from)) {
                  if constexpr (alignof(TO) % 32 == 0) {
                     LANGULUS_SIMD_VERBOSE("Storing 256f to aligned");
                     simde_mm256_store_ps(&GetFirst(to), from);
                  }
                  else {
                     LANGULUS_SIMD_VERBOSE("Storing 256f to unaligned");
                     simde_mm256_storeu_ps(&GetFirst(to), from);
                  }
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing 256f to partial");
                  alignas(32) T temp[CountOf<R>];
                  simde_mm256_store_ps(temp, from);
                  memcpy(&GetFirst(to), temp, sizeof(to));
               }
            }
            else if constexpr (CT::Double<T>) {
               // To double array                                       
               if constexpr (sizeof(to) == sizeof(from)) {
                  if constexpr (alignof(TO) % 32 == 0) {
                     LANGULUS_SIMD_VERBOSE("Storing 256d to aligned");
                     simde_mm256_store_pd(&GetFirst(to), from);
                  }
                  else {
                     LANGULUS_SIMD_VERBOSE("Storing 256d to unaligned");
                     simde_mm256_storeu_pd(&GetFirst(to), from);
                  }
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing 256d to partial");
                  alignas(32) T temp[CountOf<R>];
                  simde_mm256_store_pd(temp, from);
                  memcpy(&GetFirst(to), temp, sizeof(to));
               }
            }
            else if constexpr (CT::Integer<T>) {
               // To integer array                                      
               if constexpr (sizeof(to) == sizeof(from)) {
                  if constexpr (alignof(TO) % 32 == 0) {
                     LANGULUS_SIMD_VERBOSE("Storing 256i to aligned");
                     simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(&GetFirst(to)), from);
                  }
                  else {
                     LANGULUS_SIMD_VERBOSE("Storing 256i to unaligned");
                     simde_mm256_storeu_si256(&GetFirst(to), from);
                  }
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing 256i to partial");
                  alignas(32) T temp[CountOf<R>];
                  simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(temp), from);
                  memcpy(&GetFirst(to), temp, sizeof(to));
               }
            }
            else LANGULUS_ERROR("Unsupported output");
         }
         else if constexpr (CT::SIMD512<R>) {
            if constexpr (CT::Float<T>) {
               // To float array                                        
               if constexpr (sizeof(to) == sizeof(from)) {
                  if constexpr (alignof(TO) % 64 == 0) {
                     LANGULUS_SIMD_VERBOSE("Storing 512f to aligned");
                     simde_mm512_store_ps(&GetFirst(to), from);
                  }
                  else {
                     LANGULUS_SIMD_VERBOSE("Storing 512f to unaligned");
                     simde_mm512_storeu_ps(&GetFirst(to), from);
                  }
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing 512f to partial");
                  alignas(64) T temp[CountOf<R>];
                  simde_mm512_store_ps(temp, from);
                  memcpy(&GetFirst(to), temp, sizeof(to));
               }
            }
            else if constexpr (CT::Double<T>) {
               // To double array                                       
               if constexpr (sizeof(to) == sizeof(from)) {
                  if constexpr (alignof(TO) % 64 == 0) {
                     LANGULUS_SIMD_VERBOSE("Storing 512d to aligned");
                     simde_mm512_store_pd(&GetFirst(to), from);
                  }
                  else {
                     LANGULUS_SIMD_VERBOSE("Storing 512d to unaligned");
                     simde_mm512_storeu_pd(&GetFirst(to), from);
                  }
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing 512d to partial");
                  alignas(64) T temp[CountOf<R>];
                  simde_mm512_store_pd(temp, from);
                  memcpy(&GetFirst(to), temp, sizeof(to));
               }
            }
            else if constexpr (CT::Integer<T>) {
               // To integer array                                      
               if constexpr (sizeof(to) == sizeof(from)) {
                  if constexpr (alignof(TO) % 64 == 0) {
                     LANGULUS_SIMD_VERBOSE("Storing 512i to aligned");
                     simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(&GetFirst(to)), from);
                  }
                  else {
                     LANGULUS_SIMD_VERBOSE("Storing 512i to unaligned");
                     simde_mm512_storeu_si512(&GetFirst(to), from);
                  }
               }
               else {
                  LANGULUS_SIMD_VERBOSE("Storing 512i to partial");
                  alignas(64) T temp[CountOf<R>];
                  simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(temp), from);
                  memcpy(&GetFirst(to), temp, sizeof(to));
               }
            }
            else LANGULUS_ERROR("Unsupported output");
         }
         else LANGULUS_ERROR("Unsupported register");
      }

      /// Fallback store routine, doesn't use SIMD, hopefully constexpr       
      ///   @param from - scalar/vector/bitmask to store                      
      ///   @param to - scalar/vector/bitmask to write                        
      LANGULUS(INLINED)
      constexpr void StoreConstexpr(const CT::NotSIMD auto& from, CT::NotSIMD auto& to) noexcept {
         using FROM = Deref<decltype(from)>;
         using TO   = Deref<decltype(to)>;
         using E    = TypeOf<TO>;
         constexpr auto S = OverlapCounts<FROM, TO>();
         static_assert(S > 0);

         if constexpr (CT::Bitmask<FROM>) {
            // Extract from bitmask                                     
            if constexpr (CT::Bitmask<TO>) {
               // Store in another bitmask                              
               to = from;
            }
            else if constexpr (CT::Vector<TO>) {
               // Store each bit into an array of different type        
               if constexpr (CT::Bool<E>) {
                  // Convert each bit to a boolean inside an array      
                  for (Offset i = 0; i < S; ++i)
                     to[i] = from[i];
               }
               else LANGULUS_ERROR("Bad output to store a bitmask");
            }
            else if constexpr (CT::Scalar<TO>) {
               if constexpr (CT::Bool<E>) {
                  // Collapse the entire bitmask to a single boolean    
                  GetFirst(to) = from;
               }
               else LANGULUS_ERROR("Bad output to store a bitmask");
            }
            else LANGULUS_ERROR("Unsupported destination");
         }
         else if constexpr (CT::Vector<FROM>) {
            // Extract from any range                                   
            if constexpr (CT::Bitmask<TO>) {
               // Store as bits inside a bitmask                        
               for (Offset i = 0; i < S; ++i)
                  to[i] = static_cast<bool>(from[i]);
            }
            else if constexpr (CT::Vector<TO>) {
               // Fill a vector, element by element                     
               for (Offset i = 0; i < S; ++i)
                  to[i] = from[i];
            }
            else if constexpr (CT::Scalar<TO>) {
               // Store as a scalar                                     
               if constexpr (CT::Bool<E>) {
                  // Collect all booleans, so that we don't lose the    
                  // information. Short-circuit on the first falsum     
                  for (auto& it : from) {
                     if (not it) {
                        GetFirst(to) = false;
                        return;
                     }
                  }
                  GetFirst(to) = true;
               }
               else GetFirst(to) = from;
            }
            else LANGULUS_ERROR("Unsupported destination");
         }
         else if constexpr (CT::Scalar<FROM>) {
            // Extract from a scalar                                    
            if constexpr (CT::Vector<TO>) {
               // Multicast to a vector output                          
               for (Offset i = 0; i < S; ++i)
                  to[i] = from;
            }
            else GetFirst(to) = from;
         }
         else LANGULUS_ERROR("Unsupported source");
      }

   } // namespace Langulus::SIMD::Inner


   /// Generalized store routine, will attempt constexpr and SIMD execution   
   ///   @param from - what to store                                          
   ///   @param to - where to store it                                        
   LANGULUS(INLINED)
   constexpr void Store(const CT::NotSemantic auto& from, CT::NotSIMD auto& to) noexcept {
      if constexpr (CT::SIMD<decltype(from)>)
         Inner::StoreSIMD(from, to);
      else
         Inner::StoreConstexpr(from, to);
   }

} // namespace Langulus::SIMD


///                                                                           
#define LANGULUS_SIMD_ARITHMETHIC_API(OP) \
   template<class LHS, class RHS, CT::NotSemantic OUT> LANGULUS(INLINED) \
   constexpr void OP(const LHS& lhs, const RHS& rhs, OUT& out) noexcept { \
      IF_CONSTEXPR() { \
         Store(Inner::OP##Constexpr<OUT>(DesemCast(lhs), DesemCast(rhs)), out); \
      } \
      else if constexpr (CT::SIMD<OUT>) \
         out = Inner::OP<OUT>(lhs, rhs); \
      else \
         Store(Inner::OP<OUT>(DesemCast(lhs), DesemCast(rhs)), out); \
   } \
   template<class LHS, class RHS, CT::NotSemantic OUT = LosslessArray<LHS, RHS>> \
   NOD() LANGULUS(INLINED) \
   constexpr auto OP(const LHS& lhs, const RHS& rhs) noexcept { \
      OUT out; \
      OP(DesemCast(lhs), DesemCast(rhs), out); \
      if constexpr (CT::Similar<LHS, RHS> or CT::DerivedFrom<LHS, RHS>) \
         return LHS {out}; \
      else if constexpr (CT::DerivedFrom<RHS, LHS>) \
         return RHS {out}; \
      else \
         return out; \
   }

///                                                                           
#define LANGULUS_SIMD_ARITHMETHIC_UNARY_API(OP) \
   template<class VAL, CT::NotSemantic OUT> LANGULUS(INLINED) \
   constexpr void OP(const VAL& val, OUT& out) noexcept { \
      IF_CONSTEXPR() { \
         Store(Inner::OP##Constexpr<OUT>(DesemCast(val)), out); \
      } \
      else if constexpr (CT::SIMD<OUT>) \
         out = Inner::OP<OUT>(val); \
      else \
         Store(Inner::OP<OUT>(DesemCast(val)), out); \
   } \
   template<class VAL, CT::NotSemantic OUT = LosslessArray<VAL>> \
   NOD() LANGULUS(INLINED) \
   constexpr auto OP(const VAL& val) noexcept { \
      OUT out; \
      OP(DesemCast(val), out); \
      return out; \
   }
