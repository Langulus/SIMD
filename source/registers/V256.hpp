#pragma once
#include "../Common.hpp"
#include "V128.hpp"


namespace Langulus::SIMD
{

   ///                                                                        
   /// 256bit register of single-precision real numbers                       
   ///                                                                        
   template<>
   struct V256<simde_float32> {
      using CTTI_InnerType = simde_float32;
      static constexpr int CTTI_SIMD_Trait = 256;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float32);

      simde__m256 m;

      V256() noexcept = default;

      LANGULUS(INLINED)
      V256(const simde__m256& v) noexcept
         : m {v} {}

      /// Make a register by combining the first 4 elements of lo with the    
      /// first 4 element of hi                                               
      LANGULUS(INLINED)
      V256(const simde__m256& lo, const simde__m256& hi) noexcept
         : m {simde_mm256_permute2f128_ps(lo, hi, Shuffle4(0, 2))} {}

      /// Create a zero-filled register of this kind                          
      LANGULUS(INLINED)
      static V256 Zero() noexcept {
         return simde_mm256_setzero_ps();
      }

      /// Create a 1-bit-filled register of this kind                         
      LANGULUS(INLINED)
      static V256 Full() noexcept {
         return simde_mm256_castsi256_ps(simde_mm256_set1_epi32(-1));
      }

      LANGULUS(INLINED)
      operator simde__m256& () noexcept {
         return m;
      }

      LANGULUS(INLINED)
      operator simde__m256 const& () const noexcept {
         return m;
      }

      /// Cast to a smaller register                                          
      explicit operator V128<simde_float32>() const noexcept {
         return simde_mm256_castps256_ps128(m);
      }

   #if LANGULUS_SIMD(512BIT)
      /// Cast to a bigger register                                           
      explicit operator V512<simde_float32>() const noexcept;
   #endif

      /// Bitwise not operator                                                
      LANGULUS(INLINED)
      V256 operator ! () const noexcept {
         return simde_mm256_xor_ps(m, Full());
      }
   };


   LANGULUS(INLINED)
   V128<simde_float32>::operator V256<simde_float32>() const noexcept {
      return simde_mm256_castps128_ps256(m);
   }


   ///                                                                        
   /// 256bit register of double-precision real numbers                       
   ///                                                                        
   template<>
   struct V256<simde_float64> {
      using CTTI_InnerType = simde_float64;
      static constexpr int CTTI_SIMD_Trait = 256;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float64);

      simde__m256d m;

      V256() noexcept = default;

      LANGULUS(INLINED)
      V256(const simde__m256d& v) noexcept
         : m {v} {}

      /// Make a register by combining the first 2 elements of lo with the    
      /// first 2 element of hi                                               
      LANGULUS(INLINED)
      V256(const simde__m256d& lo, const simde__m256d& hi) noexcept
         : m {simde_mm256_permute2f128_pd(lo, hi, Shuffle4(0, 2))} {}

      /// Create a zero-filled register of this kind                          
      LANGULUS(INLINED)
      static V256 Zero() noexcept {
         return simde_mm256_setzero_pd();
      }

      /// Create a 1-bit-filled register of this kind                         
      LANGULUS(INLINED)
      static V256 Full() noexcept {
         return simde_mm256_castsi256_pd(simde_mm256_set1_epi32(-1));
      }

      LANGULUS(INLINED)
      operator simde__m256d& () noexcept {
         return m;
      }

      LANGULUS(INLINED)
      operator simde__m256d const& () const noexcept {
         return m;
      }

      /// Cast to a smaller register                                          
      explicit operator V128<simde_float64>() const noexcept {
         return simde_mm256_castpd256_pd128(m);
      }

   #if LANGULUS_SIMD(512BIT)
      /// Cast to a bigger register                                           
      explicit operator V512<simde_float64>() const noexcept;
   #endif

      /// Bitwise not operator                                                
      LANGULUS(INLINED)
      V256 operator ! () const noexcept {
         return simde_mm256_xor_pd(m, Full());
      }
   };
   

   LANGULUS(INLINED)
   V128<simde_float64>::operator V256<simde_float64>() const noexcept {
      return simde_mm256_castpd128_pd256(m);
   }


   ///                                                                        
   /// 256bit register of any integer                                         
   ///                                                                        
   template<IntElement T>
   struct V256<T> {
      using CTTI_InnerType = T;
      static constexpr int CTTI_SIMD_Trait = 256;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(T);

      simde__m256i m;

      V256() noexcept = default;

      LANGULUS(INLINED)
      V256(const simde__m256i& v) noexcept
         : m {v} {}

      /// Make a register by combining the lower elements of lo with the      
      /// lower elements of hi                                                
      LANGULUS(INLINED)
      V256(const simde__m256i& lo, const simde__m256i& hi) noexcept
         : m {simde_mm256_permute2f128_si256(lo, hi, Shuffle4(0, 2))} {}

      /// Make a register by combining all elements of lo with all elements   
      /// of hi                                                               
      LANGULUS(INLINED)
      V256(const simde__m128i& lo, const simde__m128i& hi) noexcept
         : m {simde_mm256_permute2f128_si256(
            simde_mm256_castsi128_si256(lo),
            simde_mm256_castsi128_si256(hi),
            Shuffle4(0, 2)
         )} {}

      /// Create a zero-filled register of this kind                          
      LANGULUS(INLINED)
      static V256 Zero() noexcept {
         return simde_mm256_setzero_si256();
      }

      /// Create a 1-bit-filled register of this kind                         
      LANGULUS(INLINED)
      static V256 Full() noexcept {
         return simde_mm256_set1_epi32(-1);
      }

      LANGULUS(INLINED)
      operator simde__m256i& () noexcept {
         return m;
      }

      LANGULUS(INLINED)
      operator simde__m256i const& () const noexcept {
         return m;
      }

      /// Cast to a smaller register                                          
      explicit operator V128<T>() const noexcept {
         return simde_mm256_castsi256_si128(m);
      }

   #if LANGULUS_SIMD(512BIT)
      /// Cast to a bigger register                                           
      explicit operator V512<T>() const noexcept;
   #endif

      /// Unpack lower half of elements of T to a wider type                  
      LANGULUS(INLINED)
      auto UnpackLo() const noexcept -> V256<Wider<T>> {
         if constexpr (CT::SignedInteger8<T>)
            return simde_mm256_cvtepi8_epi16 (simde_mm256_castsi256_si128(m));
         else if constexpr (CT::UnsignedInteger8<T>)
            return simde_mm256_cvtepu8_epi16 (simde_mm256_castsi256_si128(m));
         else if constexpr (CT::SignedInteger16<T>)
            return simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(m));
         else if constexpr (CT::UnsignedInteger16<T>)
            return simde_mm256_cvtepu16_epi32(simde_mm256_castsi256_si128(m));
         else if constexpr (CT::SignedInteger32<T>)
            return simde_mm256_cvtepi32_epi64(simde_mm256_castsi256_si128(m));
         else if constexpr (CT::UnsignedInteger32<T>)
            return simde_mm256_cvtepu32_epi64(simde_mm256_castsi256_si128(m));
         else
            static_assert(false, "Can't unpack this type");
      }

      /// Unpack higher half of elements of T to a wider type                 
      LANGULUS(INLINED)
      auto UnpackHi() const noexcept -> V256<Wider<T>> {
         if constexpr (CT::SignedInteger8<T>)
            return simde_mm256_cvtepi8_epi16 (simde_mm256_extractf128_si256(m, 1));
         else if constexpr (CT::UnsignedInteger8<T>)
            return simde_mm256_cvtepu8_epi16 (simde_mm256_extractf128_si256(m, 1));
         else if constexpr (CT::SignedInteger16<T>)
            return simde_mm256_cvtepi16_epi32(simde_mm256_extractf128_si256(m, 1));
         else if constexpr (CT::UnsignedInteger16<T>)
            return simde_mm256_cvtepu16_epi32(simde_mm256_extractf128_si256(m, 1));
         else if constexpr (CT::SignedInteger32<T>)
            return simde_mm256_cvtepi32_epi64(simde_mm256_extractf128_si256(m, 1));
         else if constexpr (CT::UnsignedInteger32<T>)
            return simde_mm256_cvtepu32_epi64(simde_mm256_extractf128_si256(m, 1));
         else
            static_assert(false, "Can't unpack this type");
      }

      /// Pack all elements of this register into a narrower type, filling the
      /// lower half of the resulting register                                
      LANGULUS(INLINED)
      auto Pack() const noexcept -> V256<Narrower<T>> {
         if constexpr (CT::SignedInteger16<T>) {
            const auto lo_lane = simde_mm256_castsi256_si128(m);
            const auto hi_lane = simde_mm256_extracti128_si256(m, 1);
            return simde_mm256_castsi128_si256(simde_mm_packs_epi16(lo_lane, hi_lane));
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            const auto lo_lane = simde_mm256_castsi256_si128(m);
            const auto hi_lane = simde_mm256_extracti128_si256(m, 1);
            return simde_mm256_castsi128_si256(simde_mm_packus_epi16(lo_lane, hi_lane));
         }
         else if constexpr (CT::SignedInteger32<T>)
            return simde_mm256_packs_epi32 (m, simde_mm256_permute2x128_si256(m, m, 1));
         else if constexpr (CT::UnsignedInteger32<T>)
            return simde_mm256_packus_epi32(m, simde_mm256_permute2x128_si256(m, m, 1));
         else if constexpr (CT::Integer64<T>) {
            #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
               return simde_mm256_cvtepi64_epi32(m);
            #else
               // Grab the 32-bit low halves of 64-bit elements         
               auto combined = simde_mm256_shuffle_ps(
                  simde_mm256_castsi256_ps(m),
                  simde_mm256_castsi256_ps(m),
                  SIMDE_MM_SHUFFLE(2, 0, 2, 0)
               );

               // {b3,b2, a3,a2 | b1,b0, a1,a0}  from high to low       
               // Re-arrange pairs of 32-bit elements with vpermpd      
               // (or vpermq if you want)                               
               auto ordered = simde_mm256_permute4x64_pd(
                  simde_mm256_castps_pd(combined),
                  SIMDE_MM_SHUFFLE(3, 1, 2, 0)
               );

               return simde_mm256_castpd_si256(ordered);
            #endif
         }
         else static_assert(false, "Can't pack this type");
      }

      /// Bitwise not operator                                                
      LANGULUS(INLINED)
      V256 operator ! () const noexcept {
         return simde_mm256_xor_si256(m, Full());
      }
   };


   template<IntElement T> LANGULUS(INLINED)
   V128<T>::operator V256<T>() const noexcept {
      return simde_mm256_castsi128_si256(m);
   }


   LANGULUS(INLINED)
   V256f _mm_halfflip(const V256f what) noexcept {
      return {simde_mm256_permute2f128_ps(what.m, what.m, 0x20)};
   }

   LANGULUS(INLINED)
   V256d _mm_halfflip(const V256d what) noexcept {
      return {simde_mm256_permute2f128_pd(what.m, what.m, 0x20)};
   }

   template<CT::Integer T> LANGULUS(INLINED)
   V256<T> _mm_halfflip(const V256<T> what) noexcept {
      return {simde_mm256_permute2x128_si256(what.m, what.m, 1)};
   }

   LANGULUS(INLINED)
   simde__m256i lgls_blendv_epi32(simde__m256i a, simde__m256i b, simde__m256i mask) {
      return simde_mm256_castps_si256(simde_mm256_blendv_ps(
         simde_mm256_castsi256_ps(a),
         simde_mm256_castsi256_ps(b),
         simde_mm256_castsi256_ps(mask)
      ));
   }

   /// Pack 16bit integers (signed or not) to 8bit integers                   
   ///   @tparam SATURATE - true to saturate, false to truncate               
   ///   @param low - lower sixteen 16bit integers                            
   ///   @param high - higher sixteen 16bit integers                          
   ///   @return the combined 32 truncated/saturated 8bit equivalents         
   template<bool SATURATE, CT::Integer16 T> LANGULUS(INLINED)
   auto lgls_pack_epi16(V256<T> low, V256<T> high) {
      if constexpr (SATURATE) {
         if constexpr (CT::Signed<T>)
            return V256i8 {simde_mm256_permute4x64_epi64(simde_mm256_packs_epi16(low, high), Shuffle2(0,2,1,3))};
         else
            return V256u8 {simde_mm256_permute4x64_epi64(simde_mm256_packus_epi16(low, high), Shuffle2(0,2,1,3))};
      }
      else {
         #if LANGULUS_SIMD(512BIT)
            const auto r = simde_mm256_or_si256(
               simde_mm256_cvtepi16_epi8(low),
               _mm_halfflip(simde_mm256_cvtepi16_epi8(high))
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

            const auto C1 = simde_mm_or_si128(
               simde_mm_shuffle_epi8(simde_mm256_extracti128_si256(low, 0), maskLo),
               simde_mm_shuffle_epi8(simde_mm256_extracti128_si256(low, 1), maskHi)
            );
            const auto C2 = simde_mm_or_si128(
               simde_mm_shuffle_epi8(simde_mm256_extracti128_si256(high, 0), maskLo),
               simde_mm_shuffle_epi8(simde_mm256_extracti128_si256(high, 1), maskHi)
            );

            const auto C = simde_mm256_inserti128_si256(simde_mm256_setzero_si256(), C1, 0);
            const auto r = simde_mm256_inserti128_si256(C, C2, 1);
         #endif

         if constexpr (CT::Signed<T>)
            return V256i8 {r};
         else
            return V256u8 {r};
      }
   }

   /// Pack 32bit integers (signed or not) to 16bit integers                  
   ///   @tparam SATURATE - true to saturate, false to truncate               
   ///   @param low - lower eight 32bit integers                              
   ///   @param high - higher eight 32bit integers                            
   ///   @return the combined 16 truncated/saturated 16bit equivalents        
   template<bool SATURATE, CT::Integer32 T> LANGULUS(INLINED)
   auto lgls_pack_epi32(V256<T> low, V256<T> high) {
      if constexpr (SATURATE) {
         if constexpr (CT::Signed<T>)
            return V256i16 {simde_mm256_permute4x64_epi64(simde_mm256_packs_epi32(low, high), Shuffle2(0,2,1,3))};
         else
            return V256u16 {simde_mm256_permute4x64_epi64(simde_mm256_packus_epi32(low, high), Shuffle2(0,2,1,3))};
      }
      else {
         #if LANGULUS_SIMD(512BIT)
            const auto r = simde_mm_or_si128(
               simde_mm_cvtepi32_epi16(low), 
               _mm_halfflip(simde_mm_cvtepi32_epi16(high))
            );
         #else
            const auto maskLo = simde_mm_set_epi8(
               -1, -1, -1, -1, -1, -1, -1, -1,
               13, 12, 9, 8, 5, 4, 1, 0
            );
            const auto maskHi = simde_mm_set_epi8(
               13, 12, 9, 8, 5, 4, 1, 0,
               -1, -1, -1, -1, -1, -1, -1, -1
            );

            const auto C1 = simde_mm_or_si128(
               simde_mm_shuffle_epi8(simde_mm256_extracti128_si256(low, 0), maskLo),
               simde_mm_shuffle_epi8(simde_mm256_extracti128_si256(low, 1), maskHi)
            );
            const auto C2 = simde_mm_or_si128(
               simde_mm_shuffle_epi8(simde_mm256_extracti128_si256(high, 0), maskLo),
               simde_mm_shuffle_epi8(simde_mm256_extracti128_si256(high, 1), maskHi)
            );

            auto C = simde_mm256_inserti128_si256(simde_mm256_setzero_si256(), C1, 0);
            const auto r = simde_mm256_inserti128_si256(C, C2, 1);
         #endif

         if constexpr (CT::Signed<T>)
            return V256i16 {r};
         else
            return V256u16 {r};
      }
   }

   /// Pack 64bit integers (signed or not) to 32bit integers with truncation  
   /// https://stackoverflow.com/questions/69408063                           
   ///   @tparam SATURATE - true to saturate, false to truncate               
   ///   @param a - lower four 64bit integers                                 
   ///   @param b - higher four 64bit integers                                
   ///   @return the combined 4 truncated 32bit equivalents                   
   template<CT::Integer64 T> LANGULUS(INLINED)
   auto lgls_pack_epi64(V256<T> a, V256<T> b) {
      #if LANGULUS_SIMD(512BIT)
         const auto r = _mm256_cvtepi64_epi32(a, b);
      #else
         // Grab the 32-bit low halves of 64-bit elements into one vector
         auto combined = _mm256_shuffle_ps(
            _mm256_castsi256_ps(a.m),
            _mm256_castsi256_ps(b.m),
            _MM_SHUFFLE(2, 0, 2, 0)
         );

         // {b3,b2, a3,a2 | b1,b0, a1,a0}  from high to low             
         // Re-arrange pairs of 32-bit elements with vpermpd            
         // (or vpermq if you want)                                     
         auto ordered = _mm256_permute4x64_pd(
            _mm256_castps_pd(combined),
            _MM_SHUFFLE(3, 1, 2, 0)
         );

         const auto r = _mm256_castpd_si256(ordered);
      #endif

      if constexpr (CT::Signed<T>)
         return V256i32 {r};
      else
         return V256u32 {r};
   }

} // namespace Langulus::SIMD

namespace Langulus::CT
{
   /// Concept for 256bit SIMD float registers                                
   template<class...T>
   concept SIMD256f = ((Deref<T>::CTTI_SIMD_Trait == 256
       and CT::Float<TypeOf<T>>) and ...);

   /// Concept for 256bit SIMD double registers                               
   template<class...T>
   concept SIMD256d = ((Deref<T>::CTTI_SIMD_Trait == 256
       and CT::Double<TypeOf<T>>) and ...);

   /// Concept for 256bit SIMD integer/bool registers                         
   template<class...T>
   concept SIMD256i = ((Deref<T>::CTTI_SIMD_Trait == 256
       and CT::Integer<TypeOf<T>>) and ...);

   /// Concept for 256bit SIMD registers                                      
   template<class...T>
   concept SIMD256  = ((Deref<T>::CTTI_SIMD_Trait == 256) and ...);

} // namespace Langulus::CT