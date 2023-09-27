///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include <RTTI/MetaData.hpp>
#include <array>

// The following settings:                                              
//    -G "Visual Studio 17 2022" -A Win32 -T "ClangCl"                  
//    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="/arch:SSE /EHsc"    
// Cause:                                                               
//    simde-f16.h(94,11): error : _Float16 is not supported             
// So make sure _Float16 identifier exists in such cases, and if not,   
// use the portable alternative provided by SIMDe                       
#ifdef __is_identifier
  #if !__is_identifier(_Float16)
    #define SIMDE_FLOAT16_API 1     // SIMDE_FLOAT16_API_PORTABLE == 1  
  #endif
#endif

#if LANGULUS_ALIGNMENT >= 64
   #include <simde/x86/avx512.h>
#endif

#if LANGULUS_ALIGNMENT >= 32
   #include <simde/x86/avx2.h>
   #include <simde/x86/avx.h>
#endif

#if LANGULUS_ALIGNMENT >= 16
   #include <simde/x86/sse4.2.h>
   #include <simde/x86/sse4.1.h>
   #include <simde/x86/ssse3.h>
   #include <simde/x86/sse3.h>
   #include <simde/x86/sse2.h>
   #include <simde/x86/sse.h>
   #include <simde/x86/svml.h>
#endif

#if defined(LANGULUS_EXPORT_ALL) or defined(LANGULUS_EXPORT_SIMD)
   #define LANGULUS_API_SIMD() LANGULUS_EXPORT()
#else
   #define LANGULUS_API_SIMD() LANGULUS_IMPORT()
#endif

LANGULUS_EXCEPTION(DivisionByZero);

#define LANGULUS_SIMD(a) LANGULUS_SIMD_##a()

#ifndef LANGULUS_SIMD_VERBOSE
   #define LANGULUS_SIMD_VERBOSE(...)
   #define LANGULUS_SIMD_VERBOSE_TAB(...)
#else
   #undef LANGULUS_SIMD_VERBOSE
   #define LANGULUS_SIMD_VERBOSE(...) \
      Logger::Info(__VA_ARGS__)
   #define LANGULUS_SIMD_VERBOSE_TAB(...) \
      const auto scoped = Logger::Info(__VA_ARGS__, Logger::Tabs {})
#endif


///                                                                           
///   Detect available SIMD                                                   
///                                                                           
/// By default, nothing is enabled                                            
#define LANGULUS_SIMD_AVX512BW() 0
#define LANGULUS_SIMD_AVX512CD() 0
#define LANGULUS_SIMD_AVX512DQ() 0
#define LANGULUS_SIMD_AVX512F() 0
#define LANGULUS_SIMD_AVX512VL() 0
#define LANGULUS_SIMD_AVX512() 0
#define LANGULUS_SIMD_AVX2() 0
#define LANGULUS_SIMD_AVX() 0
#define LANGULUS_SIMD_SSE4_2() 0
#define LANGULUS_SIMD_SSE4_1() 0
#define LANGULUS_SIMD_SSSE3() 0
#define LANGULUS_SIMD_SSE3() 0
#define LANGULUS_SIMD_SSE2() 0
#define LANGULUS_SIMD_SSE() 0

/// Categorization based on register size                                     
#define LANGULUS_SIMD_128BIT() 0
#define LANGULUS_SIMD_256BIT() 0
#define LANGULUS_SIMD_512BIT() 0

#if defined (SIMDE_ARCH_X86_AVX512BW) and LANGULUS_ALIGNMENT >= 64
   #undef LANGULUS_SIMD_AVX512BW
   #define LANGULUS_SIMD_AVX512BW() 1
   #undef LANGULUS_SIMD_256BIT
   #define LANGULUS_SIMD_256BIT() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_AVX512CD) and LANGULUS_ALIGNMENT >= 64
   #undef LANGULUS_SIMD_AVX512CD
   #define LANGULUS_SIMD_AVX512CD() 1
   #undef LANGULUS_SIMD_256BIT
   #define LANGULUS_SIMD_256BIT() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_AVX512DQ) and LANGULUS_ALIGNMENT >= 64
   #undef LANGULUS_SIMD_AVX512DQ
   #define LANGULUS_SIMD_AVX512DQ() 1
   #undef LANGULUS_SIMD_256BIT
   #define LANGULUS_SIMD_256BIT() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_AVX512F) and LANGULUS_ALIGNMENT >= 64
   #undef LANGULUS_SIMD_AVX512F
   #define LANGULUS_SIMD_AVX512F() 1
   #undef LANGULUS_SIMD_256BIT
   #define LANGULUS_SIMD_256BIT() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_AVX512VL) and LANGULUS_ALIGNMENT >= 64
   #undef LANGULUS_SIMD_AVX512VL
   #define LANGULUS_SIMD_AVX512VL() 1
   #undef LANGULUS_SIMD_256BIT
   #define LANGULUS_SIMD_256BIT() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if LANGULUS_SIMD(AVX512BW) and LANGULUS_SIMD(AVX512CD) \
                            and LANGULUS_SIMD(AVX512DQ) \
                            and LANGULUS_SIMD(AVX512F)  \
                            and LANGULUS_SIMD(AVX512VL) \
                            and LANGULUS_ALIGNMENT >= 64
   #undef LANGULUS_SIMD_AVX512
   #define LANGULUS_SIMD_AVX512() 1
   #undef LANGULUS_SIMD_512BIT
   #define LANGULUS_SIMD_512BIT() 1
   #undef LANGULUS_SIMD_256BIT
   #define LANGULUS_SIMD_256BIT() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_AVX2) and LANGULUS_ALIGNMENT >= 32
   #undef LANGULUS_SIMD_AVX2
   #define LANGULUS_SIMD_AVX2() 1
   #undef LANGULUS_SIMD_256BIT
   #define LANGULUS_SIMD_256BIT() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_AVX) and LANGULUS_ALIGNMENT >= 32
   #undef LANGULUS_SIMD_AVX
   #define LANGULUS_SIMD_AVX() 1
   #undef LANGULUS_SIMD_256BIT
   #define LANGULUS_SIMD_256BIT() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_SSE4_2) and LANGULUS_ALIGNMENT >= 16
   #undef LANGULUS_SIMD_SSE4_2
   #define LANGULUS_SIMD_SSE4_2() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_SSE4_1) and LANGULUS_ALIGNMENT >= 16
   #undef LANGULUS_SIMD_SSE4_1
   #define LANGULUS_SIMD_SSE4_1() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_SSSE3) and LANGULUS_ALIGNMENT >= 16
   #undef LANGULUS_SIMD_SSSE3
   #define LANGULUS_SIMD_SSSE3() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_SSE3) and LANGULUS_ALIGNMENT >= 16
   #undef LANGULUS_SIMD_SSE3
   #define LANGULUS_SIMD_SSE3() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_SSE2) and LANGULUS_ALIGNMENT >= 16
   #undef LANGULUS_SIMD_SSE2
   #define LANGULUS_SIMD_SSE2() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#if defined(SIMDE_ARCH_X86_SSE) and LANGULUS_ALIGNMENT >= 16
   #undef LANGULUS_SIMD_SSE
   #define LANGULUS_SIMD_SSE() 1
   #undef LANGULUS_SIMD_128BIT
   #define LANGULUS_SIMD_128BIT() 1
#endif

#include "IgnoreWarningsPush.inl"

namespace Langulus::CT
{

   /// Concept for 128bit SIMD registers                                      
   template<class... T>
   concept SIMD128 = (ExactAsOneOf<T, simde__m128, simde__m128d, simde__m128i> and ...);

   /// Concept for 128bit SIMD float registers                                
   template<class... T>
   concept SIMD128f = (Exact<T, simde__m128> and ...);

   /// Concept for 128bit SIMD double registers                               
   template<class... T>
   concept SIMD128d = (Exact<T, simde__m128d> and ...);

   /// Concept for 128bit SIMD integer registers                              
   template<class... T>
   concept SIMD128i = (Exact<T, simde__m128i> and ...);

   /// Concept for 256bit SIMD registers                                      
   template<class... T>
   concept SIMD256 = (ExactAsOneOf<T, simde__m256, simde__m256d, simde__m256i> and ...);

   /// Concept for 256bit SIMD float registers                                
   template<class... T>
   concept SIMD256f = (Exact<T, simde__m256> and ...);

   /// Concept for 256bit SIMD double registers                               
   template<class... T>
   concept SIMD256d = (Exact<T, simde__m256d> and ...);

   /// Concept for 256bit SIMD integer registers                              
   template<class... T>
   concept SIMD256i = (Exact<T, simde__m256i> and ...);

   /// Concept for 512bit SIMD registers                                      
   template<class... T>
   concept SIMD512 = (ExactAsOneOf<T, simde__m512, simde__m512d, simde__m512i> and ...);

   /// Concept for 512bit SIMD float registers                                
   template<class... T>
   concept SIMD512f = (Exact<T, simde__m512> and ...);

   /// Concept for 512bit SIMD double registers                               
   template<class... T>
   concept SIMD512d = (Exact<T, simde__m512d> and ...);

   /// Concept for 512bit SIMD integer registers                              
   template<class... T>
   concept SIMD512i = (Exact<T, simde__m512i> and ...);

   /// Concept for SIMD registers                                             
   template<class... T>
   concept SIMD   = ((SIMD128<T> or SIMD256<T> or SIMD512<T>) and ...);

   /// Concept anything but SIMD registers                                    
   template<class... T>
   concept NotSIMD = not SIMD<T...>;

   /// Concept for detecting bitmask types                                    
   template<class... T>
   concept Bitmask = ((Decay<T>::IsBitmask) and ...);

} // namespace Langulus::CT

namespace Langulus::SIMD
{
   
   namespace Inner
   {

      template<CT::NotSIMD T>
      NOD() constexpr decltype(auto) GetFirst(const T& a) noexcept {
         if constexpr (CT::Array<T> or (CT::Dense<T> and requires { a[0]; }))
            return (a[0]);
         else
            return (a);
      }

      template<CT::NotSIMD T>
      NOD() constexpr decltype(auto) GetFirst(T& a) noexcept {
         if constexpr (CT::Array<T> or (CT::Dense<T> and requires { a[0]; }))
            return (a[0]);
         else
            return (a);
      }

   } // namespace Langulus::SIMD::Inner

   using ::Langulus::Inner::Unsupported;

   /// Got these from:                                                        
   /// https://stackoverflow.com/questions/41144668                           
   LANGULUS(INLINED)
   simde__m128d uint64_to_double_full(simde__m128i x) {
      simde__m128i xH = simde_mm_srli_epi64(x, 32);
      xH = simde_mm_or_si128(xH, simde_mm_castpd_si128(simde_mm_set1_pd(19342813113834066795298816.)));          //  2^84
      simde__m128i xL = simde_mm_blend_epi16(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
      simde__m128d f = simde_mm_sub_pd(simde_mm_castsi128_pd(xH), simde_mm_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52
      return simde_mm_add_pd(f, simde_mm_castsi128_pd(xL));
   }

   LANGULUS(INLINED)
   simde__m128d int64_to_double_full(simde__m128i x) {
      simde__m128i xH = simde_mm_srai_epi32(x, 16);
      xH = simde_mm_blend_epi16(xH, simde_mm_setzero_si128(), 0x33);
      xH = simde_mm_add_epi64(xH, simde_mm_castpd_si128(simde_mm_set1_pd(442721857769029238784.)));              //  3*2^67
      simde__m128i xL = simde_mm_blend_epi16(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)), 0x88);   //  2^52
      simde__m128d f = simde_mm_sub_pd(simde_mm_castsi128_pd(xH), simde_mm_set1_pd(442726361368656609280.));          //  3*2^67 + 2^52
      return simde_mm_add_pd(f, simde_mm_castsi128_pd(xL));
   }

   /// Only works for inputs in the range: [-2^51, 2^51]                      
   LANGULUS(INLINED)
   simde__m128d int64_to_double(simde__m128i x) {
      x = simde_mm_add_epi64(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0018000000000000)));
      return simde_mm_sub_pd(simde_mm_castsi128_pd(x), simde_mm_set1_pd(0x0018000000000000));
   }

   /// Only works for inputs in the range: [0, 2^52)                          
   LANGULUS(INLINED)
   simde__m128d uint64_to_double(simde__m128i x) {
      x = simde_mm_or_si128(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)));
      return simde_mm_sub_pd(simde_mm_castsi128_pd(x), simde_mm_set1_pd(0x0010000000000000));
   }

   /// Only works for inputs in the range: [-2^51, 2^51]                      
   LANGULUS(INLINED)
   simde__m128i double_to_int64(simde__m128d x) {
      x = simde_mm_add_pd(x, simde_mm_set1_pd(0x0018000000000000));
      return simde_mm_sub_epi64(
         simde_mm_castpd_si128(x),
         simde_mm_castpd_si128(simde_mm_set1_pd(0x0018000000000000))
      );
   }

   /// Only works for inputs in the range: [0, 2^52)                          
   LANGULUS(INLINED)
   simde__m128i double_to_uint64(simde__m128d x) {
      x = simde_mm_add_pd(x, simde_mm_set1_pd(0x0010000000000000));
      return simde_mm_xor_si128(
         simde_mm_castpd_si128(x),
         simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000))
      );
   }

   /// Shuffle eight indices                                                  
   NOD() constexpr int Shuffle(int&& z1, int&& y1, int&& x1, int&& w1, int&& z0, int&& y0, int&& x0, int&& w0) noexcept {
      // 8 indices, 4 bits each                                         
      return (z1 << 28) | (y1 << 24) | (x1 << 20) | (w1 << 16) | (z0 << 12) | (y0 << 8) | (x0 << 4) | w0;
   }

   /// Shuffle four indices                                                   
   NOD() constexpr int Shuffle(int&& z, int&& y, int&& x, int&& w) noexcept {
      // 4 indices, 2 bits each                                         
      return (z << 6) | (y << 4) | (x << 2) | w;
   }

   /// Shuffle two indices                                                    
   NOD() constexpr int Shuffle(int&& x, int&& w) noexcept {
      // 2 indices, 1 bit each                                          
      return (x << 1) | w;
   }

   ///                                                                        
   LANGULUS(INLINED)
   simde__m128 _mm_halfflip(const simde__m128& what) noexcept {
      return simde_mm_permute_ps(what, Shuffle(2, 3, 0, 1));
   }

   LANGULUS(INLINED)
   simde__m128d _mm_halfflip(const simde__m128d& what) noexcept {
      return simde_mm_permute_pd(what, Shuffle(1, 0));
   }

   LANGULUS(INLINED)
   simde__m128i _mm_halfflip(const simde__m128i& what) noexcept {
      constexpr int8_t imm8 = static_cast<int8_t>(Shuffle(0, 1, 2, 3));
      return simde_mm_shuffle_epi32(what, imm8);
   }

#if LANGULUS_SIMD(256BIT)
   LANGULUS(INLINED)
   simde__m256 _mm_halfflip(const simde__m256& what) noexcept {
      return simde_mm256_permute2f128_ps(what, what, 0x20);   // AVX
   }

   LANGULUS(INLINED)
   simde__m256d _mm_halfflip(const simde__m256d& what) noexcept {
      return simde_mm256_permute2f128_pd(what, what, 0x20);   // AVX
   }

   LANGULUS(INLINED)
   simde__m256i _mm_halfflip(const simde__m256i& what) noexcept {
      return simde_mm256_permute2x128_si256(what, what, 1);   // AVX2
   }
#endif

   /*inline simde__m512 _mm_halfflip(const simde__m512& what) noexcept {
      return _mm512_shuffle_f32x4(what, what, _MM_SHUFFLE(2, 3, 0, 1));   // AVX512F
   }

   inline simde__m512d _mm_halfflip(const simde__m512d& what) noexcept {
      return simde_mm512_shuffle_f64x2(what, what, _MM_SHUFFLE(2, 3, 0, 1));   // AVX512F
   }

   inline simde__m512i _mm_halfflip(const simde__m512i& what) noexcept {
      return simde_mm512_shuffle_i64x2(what, what, _MM_SHUFFLE(2, 3, 0, 1));   // AVX512F
   }*/

   ///                                                                        
   LANGULUS(INLINED)
   int _mm_hmax_epu8(const simde__m128i v) noexcept {
      simde__m128i vmax = v;
      vmax = simde_mm_max_epu8(vmax, simde_mm_alignr_epi8(vmax, vmax, 1)); // SSSE3 + SSE2
      vmax = simde_mm_max_epu8(vmax, simde_mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
      vmax = simde_mm_max_epu8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epu8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi8(vmax, 0); // SSE4.1
   }

   LANGULUS(INLINED)
   int _mm_hmax_epu16(const simde__m128i v) noexcept {
      simde__m128i vmax = v;
      vmax = simde_mm_max_epu16(vmax, simde_mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
      vmax = simde_mm_max_epu16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epu16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi16(vmax, 0); // SSE2
   }

   LANGULUS(INLINED)
   int _mm_hmax_epu32(const simde__m128i v) noexcept {
      simde__m128i vmax = v;
      vmax = simde_mm_max_epu32(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epu32(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi32(vmax, 0); // SSE4.1
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
   int _mm_hmax_epi8(const simde__m128i v) noexcept {
      simde__m128i vmax = v;
      vmax = simde_mm_max_epi8(vmax, simde_mm_alignr_epi8(vmax, vmax, 1)); // SSSE3 + SSE2
      vmax = simde_mm_max_epi8(vmax, simde_mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
      vmax = simde_mm_max_epi8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epi8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi8(vmax, 0); // SSE4.1
   }

   LANGULUS(INLINED)
   int _mm_hmax_epi16(const simde__m128i v) noexcept {
      simde__m128i vmax = v;
      vmax = simde_mm_max_epi16(vmax, simde_mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
      vmax = simde_mm_max_epi16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epi16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi16(vmax, 0); // SSE2
   }

   LANGULUS(INLINED)
   int _mm_hmax_epi32(const simde__m128i v) noexcept {
      simde__m128i vmax = v;
      vmax = simde_mm_max_epi32(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epi32(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi32(vmax, 0);   // SSE2
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
   simde__m128i lgls_blendv_epi32(const simde__m128i& a, const simde__m128i& b, const simde__m128i& mask) {
      return simde_mm_castps_si128(simde_mm_blendv_ps(
         simde_mm_castsi128_ps(a),
         simde_mm_castsi128_ps(b),
         simde_mm_castsi128_ps(mask)
      ));
   }

#if LANGULUS_SIMD(256BIT)
   LANGULUS(INLINED)
   simde__m256i lgls_blendv_epi32(const simde__m256i& a, const simde__m256i& b, const simde__m256i& mask) {
      return simde_mm256_castps_si256(simde_mm256_blendv_ps(
         simde_mm256_castsi256_ps(a),
         simde_mm256_castsi256_ps(b),
         simde_mm256_castsi256_ps(mask)
      ));
   }
#endif

   /// Pack 16bit integers (signed or not) to 8bit integers with truncation   
   ///   @param low - lower eight 16bit integers                              
   ///   @param high - higher eight 16bit integers                            
   ///   @return the combined 16 truncated 8bit equivalents                   
   LANGULUS(INLINED)
   simde__m128i lgls_pack_epi16(const simde__m128i& low, const simde__m128i& high) {
      #if LANGULUS_SIMD(512BIT)
         return simde_mm_or_si128(
            simde_mm_cvtepi16_epi8(low), 
            _mm_halfflip(simde_mm_cvtepi16_epi8(high))
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

         return simde_mm_or_si128(
            simde_mm_shuffle_epi8(low, maskLo),
            simde_mm_shuffle_epi8(high, maskHi)
         );
      #endif
   }

#if LANGULUS_SIMD(256BIT)
   /// Pack 16bit integers (signed or not) to 8bit integers with truncation   
   ///   @param low - lower sixteen 16bit integers                            
   ///   @param high - higher sixteen 16bit integers                          
   ///   @return the combined 32 truncated 8bit equivalents                   
   LANGULUS(INLINED)
   simde__m256i lgls_pack_epi16(const simde__m256i& low, const simde__m256i& high) {
      #if LANGULUS_SIMD(512BIT)
         return simde_mm256_or_si256(
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
         return simde_mm256_inserti128_si256(C, C2, 1);
      #endif
   }
#endif

   /*inline simde__m512i lgls_blendv_epi32(const simde__m512i& a, const simde__m512i& b, const simde__m512i& mask) {
      use _mm512_mask_blend_ instead
      return simde_mm512_castps_si512(simde_mm512_blendv_ps(
         simde_mm512_castsi512_ps(a),
         simde_mm512_castsi512_ps(b),
         simde_mm512_castsi512_ps(mask)
      ));
   }*/
   
   /// Pack 32bit integers (signed or not) to 16bit integers with truncation  
   ///   @param low - lower four 32bit integers                               
   ///   @param high - higher four 32bit integers                             
   ///   @return the combined 8 truncated 16bit equivalents                   
   LANGULUS(INLINED)
   simde__m128i lgls_pack_epi32(const simde__m128i& low, const simde__m128i& high) {
      #if LANGULUS_SIMD(512BIT)
         return simde_mm_or_si128(
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

         return simde_mm_or_si128(
            simde_mm_shuffle_epi8(low, maskLo),
            simde_mm_shuffle_epi8(high, maskHi)
         );
      #endif
   }

#if LANGULUS_SIMD(256BIT)
   /// Pack 32bit integers (signed or not) to 16bit integers with truncation  
   ///   @param low - lower eight 32bit integers                              
   ///   @param high - higher eight 32bit integers                            
   ///   @return the combined 16 truncated 16bit equivalents                  
   LANGULUS(INLINED)
   simde__m256i lgls_pack_epi32(const simde__m256i& low, const simde__m256i& high) {
      #if LANGULUS_SIMD(512BIT)
         return simde_mm_or_si128(
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
         return simde_mm256_inserti128_si256(C, C2, 1);
      #endif
   }

   /// Pack 64bit integers (signed or not) to 32bit integers with truncation  
   /// https://stackoverflow.com/questions/69408063                           
   ///   @param a - lower four 64bit integers                                 
   ///   @param b - higher four 64bit integers                                
   ///   @return the combined 4 truncated 32bit equivalents                   
   LANGULUS(INLINED)
   simde__m256i lgls_pack_epi64(simde__m256i a, simde__m256i b) {
   #if LANGULUS_SIMD(512BIT)
      return _mm256_cvtepi64_epi8(v, v);
   #else
      // Grab the 32-bit low halves of 64-bit elements into one vector  
      auto combined = _mm256_shuffle_ps(
         _mm256_castsi256_ps(a),
         _mm256_castsi256_ps(b),
         _MM_SHUFFLE(2, 0, 2, 0)
      );

      // {b3,b2, a3,a2 | b1,b0, a1,a0}  from high to low                
      // Re-arrange pairs of 32-bit elements with vpermpd               
      // (or vpermq if you want)                                        
      auto ordered = _mm256_permute4x64_pd(
         _mm256_castps_pd(combined),
         _MM_SHUFFLE(3, 1, 2, 0)
      );

      return _mm256_castpd_si256(ordered);
   #endif
   }

#endif

   /*inline simde__m512i lgls_pack_epi32(const simde__m512i& a, const simde__m512i& b, const simde__m512i& mask) {
      use _mm512_mask_blend_ instead
      return simde_mm512_castps_si512(simde_mm512_blendv_ps(
         simde_mm512_castsi512_ps(a),
         simde_mm512_castsi512_ps(b),
         simde_mm512_castsi512_ps(mask)
      ));
   }*/

   template<class F, class T>
   concept Invocable = ::std::invocable<F, T, T>;

   namespace Inner
   {
      template<class F, class T>
      constexpr auto InvocableResultInner() noexcept {
         if constexpr (CT::Nullptr<F>)
            return Unsupported {};
         else
            return ::std::invoke_result_t<F, T, T> {};
      }
   }

   template<class F, class T>
   using InvocableResult = decltype(Inner::InvocableResultInner<F, T>());

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"

/// Make the rest of the code aware, that Langulus::SIMD has been included    
#define LANGULUS_LIBRARY_SIMD() 1
