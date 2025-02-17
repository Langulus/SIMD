#pragma once
#include "../Common.hpp"


namespace Langulus::SIMD
{

   ///                                                                        
   /// 128bit register of single-precision real numbers                       
   ///                                                                        
   template<>
   struct V128<simde_float32> {
      using CTTI_InnerType = simde_float32;
      static constexpr int CTTI_SIMD_Trait = 128;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float32);

      simde__m128 m;

      V128() noexcept = default;

      LANGULUS(INLINED)
      V128(const simde__m128& v) noexcept
         : m {v} {}

      /// Make a register by combining the first two elements of lo with the  
      /// first two element of hi                                             
      LANGULUS(INLINED)
      V128(const simde__m128& lo, const simde__m128& hi) noexcept
         : m {simde_mm_movelh_ps(lo, hi)} {}

      /// Create a zero-filled register of this kind                          
      LANGULUS(INLINED)
      static V128 Zero() noexcept {
         return simde_mm_setzero_ps();
      }

      /// Create a 1-bit-filled register of this kind                         
      LANGULUS(INLINED)
      static V128 Full() noexcept {
         return simde_mm_castsi128_ps(simde_mm_set1_epi32(-1));
      }

      LANGULUS(INLINED)
      operator simde__m128& () noexcept {
         return m;
      }

      LANGULUS(INLINED)
      operator simde__m128 const& () const noexcept {
         return m;
      }

   #if LANGULUS_SIMD(256BIT)
      /// Cast to a bigger register                                           
      explicit operator V256<simde_float32>() const noexcept;
   #endif

   #if LANGULUS_SIMD(512BIT)
      /// Cast to a bigger register                                           
      explicit operator V512<simde_float32>() const noexcept;
   #endif

      /// Bitwise not operator                                                
      LANGULUS(INLINED)
      V128 operator ! () const noexcept {
         return simde_mm_xor_ps(m, Full());
      }
   };


   ///                                                                        
   /// 128bit register of double-precision real numbers                       
   ///                                                                        
   template<>
   struct V128<simde_float64> {
      using CTTI_InnerType = simde_float64;
      static constexpr int CTTI_SIMD_Trait = 128;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float64);

      simde__m128d m;

      V128() noexcept = default;

      LANGULUS(INLINED)
      V128(const simde__m128d& v) noexcept
         : m {v} {}

      /// Make a register by combining the first element of lo with the       
      /// first element of hi                                                 
      LANGULUS(INLINED)
      V128(const simde__m128d& lo, const simde__m128d& hi) noexcept
         : m {simde_mm_unpacklo_pd(lo, hi)} {}

      /// Create a zero-filled register of this kind                          
      LANGULUS(INLINED)
      static V128 Zero() noexcept {
         return simde_mm_setzero_pd();
      }

      /// Create a 1-bit-filled register of this kind                         
      LANGULUS(INLINED)
      static V128 Full() noexcept {
         return simde_mm_castsi128_pd(simde_mm_set1_epi32(-1));
      }

      LANGULUS(INLINED)
      operator simde__m128d& () noexcept {
         return m;
      }

      LANGULUS(INLINED)
      operator simde__m128d const& () const noexcept {
         return m;
      }

   #if LANGULUS_SIMD(256BIT)
      /// Cast to a bigger register                                           
      explicit operator V256<simde_float64>() const noexcept;
   #endif

   #if LANGULUS_SIMD(512BIT)
      /// Cast to a bigger register                                           
      explicit operator V512<simde_float64>() const noexcept;
   #endif

      /// Bitwise not operator                                                
      LANGULUS(INLINED)
      V128 operator ! () const noexcept {
         return simde_mm_xor_pd(m, Full());
      }
   };


   ///                                                                        
   /// 128bit register of any integer                                         
   ///                                                                        
   template<IntElement T>
   struct V128<T> {
      using CTTI_InnerType = T;
      static constexpr int CTTI_SIMD_Trait = 128;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(T);

      simde__m128i m;

      V128() noexcept = default;

      LANGULUS(INLINED)
      V128(const simde__m128i& v) noexcept
         : m {v} {}

      /// Make a register by combining the lower half of elements of lo,      
      /// with the lower half of entries of hi into a single register         
      /// [lo0][lo1][lo2][lo3][hi0][hi1][hi2][hi3]                            
      LANGULUS(INLINED)
      V128(const simde__m128i& lo, const simde__m128i& hi) noexcept
         : m {simde_mm_unpacklo_epi64(lo, hi)} {}

      /// Create a zero-filled register of this kind                          
      LANGULUS(INLINED)
      static V128 Zero() noexcept {
         return simde_mm_setzero_si128();
      }

      /// Create a 1-bit-filled register of this kind                         
      LANGULUS(INLINED)
      static V128 Full() noexcept {
         return simde_mm_set1_epi32(-1);
      }

      LANGULUS(INLINED)
      operator simde__m128i& () noexcept {
         return m;
      }

      LANGULUS(INLINED)
      operator simde__m128i const& () const noexcept {
         return m;
      }

      /// Unpack lower half of elements of T to a wider type                  
      LANGULUS(INLINED)
      auto UnpackLo() const noexcept -> V128<Wider<T>> {
         if constexpr (CT::Integer8<T>)
            return simde_mm_unpacklo_epi8 (m, Zero());
         else if constexpr (CT::Integer16<T>)
            return simde_mm_unpacklo_epi16(m, Zero());
         else if constexpr (CT::Integer32<T>)
            return simde_mm_unpacklo_epi32(m, Zero());
         else
            static_assert(false, "Can't unpack this type");
      }

      /// Unpack higher half of elements of T to a wider type                 
      LANGULUS(INLINED)
      auto UnpackHi() const noexcept -> V128<Wider<T>> {
         if constexpr (CT::Integer8<T>)
            return simde_mm_unpackhi_epi8 (m, Zero());
         else if constexpr (CT::Integer16<T>)
            return simde_mm_unpackhi_epi16(m, Zero());
         else if constexpr (CT::Integer32<T>)
            return simde_mm_unpackhi_epi32(m, Zero());
         else
            static_assert(false, "Can't unpack this type");
      }

      /// Pack all elements of this register into a narrower type, filling the
      /// lower half of the resulting register                                
      LANGULUS(INLINED)
      auto Pack() const noexcept -> V128<Narrower<T>> {
         if constexpr (CT::SignedInteger16<T>)
            return simde_mm_packs_epi16 (m, Zero());
         else if constexpr (CT::UnsignedInteger16<T>)
            return simde_mm_packus_epi16(m, Zero());
         else if constexpr (CT::SignedInteger32<T>)
            return simde_mm_packs_epi32 (m, Zero());
         else if constexpr (CT::UnsignedInteger32<T>)
            return simde_mm_packus_epi32(m, Zero());
         else if constexpr (CT::Integer64<T>) {
            #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
               return simde_mm_cvtepi64_epi32(m);
            #else
               // Grab the 32-bit low halves of 64-bit elements         
               auto combined = simde_mm_shuffle_ps(
                  simde_mm_castsi128_ps(m),
                  simde_mm_castsi128_ps(m),
                  SIMDE_MM_SHUFFLE(2, 0, 2, 0)
               );

               // {b3, b2, a3, a2 | b1, b0, a1, a0} from high to low    
               // Re-arrange pairs of 32-bit elements with vpermpd      
               // (or vpermq if you want)                               
               auto ordered = simde_mm_permute_pd(
                  simde_mm_castps_pd(combined),
                  SIMDE_MM_SHUFFLE(0, 0, 0, 1)
               );
               return simde_mm_castpd_si128(ordered);
            #endif
         }
         else static_assert(false, "Can't pack this type");
      }

   #if LANGULUS_SIMD(256BIT)
      explicit operator V256<T>() const noexcept;
   #endif

   #if LANGULUS_SIMD(512BIT)
      explicit operator V512<T>() const noexcept;
   #endif

      /// Bitwise not operator                                                
      LANGULUS(INLINED)
      V128 operator ! () const noexcept {
         return simde_mm_xor_si128(m, Full());
      }
   };


   ///                                                                        
   LANGULUS(INLINED)
   V128f _mm_halfflip(const V128f what) noexcept {
      return {simde_mm_permute_ps(what.m, Shuffle2(2, 3, 0, 1))};
   }

   LANGULUS(INLINED)
   V128d _mm_halfflip(const V128d what) noexcept {
      return {simde_mm_permute_pd(what.m, Shuffle1(1, 0))};
   }

   template<CT::Integer T> LANGULUS(INLINED)
   V128<T> _mm_halfflip(const V128<T> what) noexcept {
      return simde_mm_shuffle_epi32(what.m, Shuffle2(2, 3, 0, 1));
   }

   ///                                                                        
   LANGULUS(INLINED)
   int _mm_hmax_epu8(const V128u8 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epu8(vmax, simde_mm_alignr_epi8(vmax, vmax, 1));
      vmax = simde_mm_max_epu8(vmax, simde_mm_alignr_epi8(vmax, vmax, 2));
      vmax = simde_mm_max_epu8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(0,3,2,1)));
      vmax = simde_mm_max_epu8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(1,0,3,2)));
      return simde_mm_extract_epi8(vmax, 0);
   }

   LANGULUS(INLINED)
   int _mm_hmax_epu16(const V128u16 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epu16(vmax, simde_mm_alignr_epi8(vmax, vmax, 2));
      vmax = simde_mm_max_epu16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(0,3,2,1)));
      vmax = simde_mm_max_epu16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(1,0,3,2)));
      return simde_mm_extract_epi16(vmax, 0);
   }

   LANGULUS(INLINED)
   int _mm_hmax_epu32(const V128u32 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epu32(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(0,3,2,1)));
      vmax = simde_mm_max_epu32(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(1,0,3,2)));
      return simde_mm_extract_epi32(vmax, 0);
   }

   /*inline uint64_t _mm_hmax_epu64(const simde__m128i v) noexcept {
      simde__m128i vmax = v;
      vmax = _mm_max_epu64(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      #if LANGULUS_BITNESS() == 32
         alignas(16) uint64_t stored[2];
         simde_mm_store_si128(reinterpret_cast<simde__m128i*>(stored), v);      // SSE2
         return stored[0];
      #else
         const auto result = _mm_extract_epi64(vmax, 0); // SSE4.1
         return reinterpret_cast<const uint64_t&>(result);
      #endif
   }*/

   LANGULUS(INLINED)
   int _mm_hmax_epi8(const V128i8 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epi8(vmax, simde_mm_alignr_epi8(vmax, vmax, 1));
      vmax = simde_mm_max_epi8(vmax, simde_mm_alignr_epi8(vmax, vmax, 2));
      vmax = simde_mm_max_epi8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(0,3,2,1)));
      vmax = simde_mm_max_epi8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(1,0,3,2)));
      return simde_mm_extract_epi8(vmax, 0);
   }

   LANGULUS(INLINED)
   int _mm_hmax_epi16(const V128i16 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epi16(vmax, simde_mm_alignr_epi8(vmax, vmax, 2));
      vmax = simde_mm_max_epi16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(0,3,2,1)));
      vmax = simde_mm_max_epi16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(1,0,3,2)));
      return simde_mm_extract_epi16(vmax, 0);
   }

   LANGULUS(INLINED)
   int _mm_hmax_epi32(const V128i32 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epi32(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(0,3,2,1)));
      vmax = simde_mm_max_epi32(vmax, simde_mm_shuffle_epi32(vmax, Shuffle2(1,0,3,2)));
      return simde_mm_extract_epi32(vmax, 0);
   }

   /*inline int64_t _mm_hmax_epi64(const simde__m128i v) noexcept {
      simde__m128i vmax = v;
      vmax = _mm_max_epi64(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      #if LANGULUS_BITNESS() == 32
         alignas(16) int64_t stored[2];
         simde_mm_store_si128(reinterpret_cast<simde__m128i*>(stored), v);      // SSE2
         return stored[0];
      #else
         const auto result = _mm_extract_epi64(vmax, 0); // SSE4.1
         return reinterpret_cast<const int64_t&>(result);
      #endif
   }*/
   
   LANGULUS(INLINED)
   simde__m128i lgls_blendv_epi32(simde__m128i a, simde__m128i b, simde__m128i mask) {
      return simde_mm_castps_si128(simde_mm_blendv_ps(
         simde_mm_castsi128_ps(a),
         simde_mm_castsi128_ps(b),
         simde_mm_castsi128_ps(mask)
      ));
   }

   /// Pack 16bit integers (signed or not) to 8bit integers                   
   ///   @tparam SATURATE - true to saturate, false to truncate               
   ///   @param low - lower eight 16bit integers                              
   ///   @param high - higher eight 16bit integers                            
   ///   @return the combined 16 truncated/saturated 8bit equivalents         
   template<bool SATURATE, CT::Integer16 T> LANGULUS(INLINED)
   auto lgls_pack_epi16(V128<T> low, V128<T> high) {
      if constexpr (SATURATE) {
         if constexpr (CT::Signed<T>)
            return V128i8 {simde_mm_packs_epi16(low, high)};
         else
            return V128u8 {simde_mm_packus_epi16(low, high)};
      }
      else {
         #if LANGULUS_SIMD(512BIT)
            const auto r = simde_mm_or_si128(
               simde_mm_cvtepi16_epi8(low.m), 
               _mm_halfflip(simde_mm_cvtepi16_epi8(high.m))
            );
         #else
            const auto maskLo = simde_mm_set_epi8(
               -1, -1, -1, -1, -1, -1, -1, -1,
               14, 12, 10, 8, 6, 4, 2, 0
            );
            const auto maskHi = simde_mm_set_epi8(
               14, 12, 10, 8, 6, 4, 2, 0,
               -1, -1, -1, -1, -1, -1, -1, -1
            );

            const auto r = simde_mm_or_si128(
               simde_mm_shuffle_epi8(low.m,  maskLo),
               simde_mm_shuffle_epi8(high.m, maskHi)
            );
         #endif

         if constexpr (CT::Signed<T>)
            return V128i8 {r};
         else
            return V128u8 {r};
      }
   }

   /// Pack 32bit integers (signed or not) to 16bit integers                  
   ///   @tparam SATURATE - true to saturate, false to truncate               
   ///   @param low - lower four 32bit integers                               
   ///   @param high - higher four 32bit integers                             
   ///   @return the combined 8 truncated/saturated 16bit equivalents         
   template<bool SATURATE, CT::Integer32 T> LANGULUS(INLINED)
   auto lgls_pack_epi32(V128<T> low, V128<T> high) {
      if constexpr (SATURATE) {
         if constexpr (CT::Signed<T>)
            return V128i16 {simde_mm_packs_epi32(low, high)};
         else
            return V128u16 {simde_mm_packus_epi32(low, high)};
      }
      else {
         #if LANGULUS_SIMD(512BIT)
            const auto r = simde_mm_or_si128(
               simde_mm_cvtepi32_epi16(low),
               _mm_halfflip(simde_mm_cvtepi32_epi16(high))
            );
         #else
            const auto maskLo = simde_mm_setr_epi8(
               0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1
            );
            const auto maskHi = simde_mm_setr_epi8(
               -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13
            );

            const auto r = simde_mm_or_si128(
               simde_mm_shuffle_epi8(low, maskLo),
               simde_mm_shuffle_epi8(high, maskHi)
            );
         #endif

         if constexpr (CT::Signed<T>)
            return V128i16 {r};
         else
            return V128u16 {r};
      }
   }

} // namespace Langulus::SIMD

namespace Langulus::CT
{
   /// Concept for 128bit SIMD float registers                                
   template<class...T>
   concept SIMD128f = ((Deref<T>::CTTI_SIMD_Trait == 128
       and CT::Float<TypeOf<T>>) and ...);

   /// Concept for 128bit SIMD double registers                               
   template<class...T>
   concept SIMD128d = ((Deref<T>::CTTI_SIMD_Trait == 128
       and CT::Double<TypeOf<T>>) and ...);

   /// Concept for 128bit SIMD integer/bool registers                         
   template<class...T>
   concept SIMD128i = ((Deref<T>::CTTI_SIMD_Trait == 128
       and CT::Integer<TypeOf<T>>) and ...);

   /// Concept for 128bit SIMD registers                                      
   template<class...T>
   concept SIMD128  = ((Deref<T>::CTTI_SIMD_Trait == 128) and ...);

} // namespace Langulus::CT