///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include <RTTI/Meta.hpp>
#include <array>

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

/// Make the rest of the code aware, that Langulus::SIMD has been included    
#define LANGULUS_LIBRARY_SIMD() 1

#ifndef LANGULUS_SIMD_VERBOSE
   #define LANGULUS_SIMD_VERBOSE(...)     LANGULUS(NOOP)
   #define LANGULUS_SIMD_VERBOSE_TAB(...) LANGULUS(NOOP)
#else
   #undef LANGULUS_SIMD_VERBOSE
   #define LANGULUS_SIMD_VERBOSE(...)     Logger::Info(__VA_ARGS__)
   #define LANGULUS_SIMD_VERBOSE_TAB(...) const auto scoped = Logger::Info(__VA_ARGS__, Logger::Tabs {})
#endif


///                                                                           
///   Detect available SIMD                                                   
///                                                                           
/// By default, nothing is enabled                                            
#define LANGULUS_SIMD_ENABLED() 0
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

#if LANGULUS_SIMD(128BIT) or LANGULUS_SIMD(256BIT) or LANGULUS_SIMD(512BIT)
   #undef LANGULUS_SIMD_ENABLED
   #define LANGULUS_SIMD_ENABLED()  1
   #define IF_LANGULUS_SIMD(a)      a
   #define IF_NOT_LANGULUS_SIMD(a)  LANGULUS(NOOP)
#else
   #define IF_LANGULUS_SIMD(a)      LANGULUS(NOOP)
   #define IF_NOT_LANGULUS_SIMD(a)  a
#endif


///                                                                           
///   Register concepts and representations                                   
///                                                                           
/// Notice how we don't use simde__m128i, simde__m256i and simde__m512i       
/// These are forbidden in langulus, because they cause type-erasure.         
/// Instead, thes are contained inside these aggregate types:                 
///   V128i<integer or bool>                                                  
///   V256i<integer or bool>                                                  
///   V512i<integer or bool>                                                  
///                                                                           
namespace Langulus::SIMD
{

   using ::Langulus::Inner::Unsupported;

   /// Single real element inside a register                                  
   template<class...T>
   concept RealElement = ((CT::ExactAsOneOf<T,
      simde_float32, simde_float64
   >) and ...);

   /// Single integer element inside a register                               
   template<class...T>
   concept IntElement = ((CT::ExactAsOneOf<T,
      ::std::int8_t,  ::std::int16_t,  ::std::int32_t,  ::std::int64_t,
      ::std::uint8_t, ::std::uint16_t, ::std::uint32_t, ::std::uint64_t,
      char8_t, char16_t, char32_t, wchar_t, Langulus::Byte
   >) and ...);

   /// Single element inside a register                                       
   template<class...T>
   concept Element = RealElement<T...> or IntElement<T...>;

#if LANGULUS_SIMD(128BIT)
   /// 128bit register                                                        
   template<class>
   struct V128;

   template<>
   struct V128<simde_float32> {
      LANGULUS(TYPED) simde_float32;
      static constexpr int CTTI_SIMD_Trait = 128;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float32);

      simde__m128 m;

      LANGULUS(INLINED)
      V128(const simde__m128& v) noexcept
         : m {v} {}

      NOD() LANGULUS(INLINED)
      static V128 Zero() noexcept {
         return simde_mm_setzero_ps();
      }
      LANGULUS(INLINED)
      operator simde__m128& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m128 const& () const noexcept {
         return m;
      }
   };

   template<>
   struct V128<simde_float64> {
      LANGULUS(TYPED) simde_float64;
      static constexpr int CTTI_SIMD_Trait = 128;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float64);

      simde__m128d m;

      LANGULUS(INLINED)
      V128(const simde__m128d& v) noexcept
         : m {v} {}

      NOD() LANGULUS(INLINED)
      static V128 Zero() noexcept {
         return simde_mm_setzero_pd();
      }
      LANGULUS(INLINED)
      operator simde__m128d& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m128d const& () const noexcept {
         return m;
      }
   };

   template<IntElement T>
   struct V128<T> {
      LANGULUS(TYPED) T;
      static constexpr int CTTI_SIMD_Trait = 128;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(T);

      simde__m128i m;

      LANGULUS(INLINED)
      V128(const simde__m128i& v) noexcept
         : m {v} {}

      NOD() LANGULUS(INLINED)
      static V128 Zero() noexcept {
         return simde_mm_setzero_si128();
      }
      LANGULUS(INLINED)
      operator simde__m128i& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m128i const& () const noexcept {
         return m;
      }

      NOD() LANGULUS(INLINED)
      auto UnpackLo() const noexcept {
         if constexpr (CT::SignedInteger8<T>)
            return V128<std::int16_t>  {simde_mm_unpacklo_epi8 (m, Zero())};
         else if constexpr (CT::UnsignedInteger8<T>)
            return V128<std::uint16_t> {simde_mm_unpacklo_epi8 (m, Zero())};
         else if constexpr (CT::SignedInteger16<T>)
            return V128<std::int32_t>  {simde_mm_unpacklo_epi16(m, Zero())};
         else if constexpr (CT::UnsignedInteger16<T>)
            return V128<std::uint32_t> {simde_mm_unpacklo_epi16(m, Zero())};
         else if constexpr (CT::SignedInteger32<T>)
            return V128<std::int64_t>  {simde_mm_unpacklo_epi32(m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V128<std::uint64_t> {simde_mm_unpacklo_epi32(m, Zero())};
         else
            LANGULUS_ERROR("Can't unpack this type");
      }

      NOD() LANGULUS(INLINED)
      auto UnpackHi() const noexcept {
         if constexpr (CT::SignedInteger8<T>)
            return V128<std::int16_t>  {simde_mm_unpackhi_epi8 (m, Zero())};
         else if constexpr (CT::UnsignedInteger8<T>)
            return V128<std::uint16_t> {simde_mm_unpackhi_epi8 (m, Zero())};
         else if constexpr (CT::SignedInteger16<T>)
            return V128<std::int32_t>  {simde_mm_unpackhi_epi16(m, Zero())};
         else if constexpr (CT::UnsignedInteger16<T>)
            return V128<std::uint32_t> {simde_mm_unpackhi_epi16(m, Zero())};
         else if constexpr (CT::SignedInteger32<T>)
            return V128<std::int64_t>  {simde_mm_unpackhi_epi32(m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V128<std::uint64_t> {simde_mm_unpackhi_epi32(m, Zero())};
         else
            LANGULUS_ERROR("Can't unpack this type");
      }

      NOD() LANGULUS(INLINED)
      auto Pack() const noexcept {
         if constexpr (CT::Integer8<T>)
            return *this;
         else if constexpr (CT::SignedInteger16<T>)
            return V128<std::int8_t>      {simde_mm_packs_epi16 (m, Zero())};
         else if constexpr (CT::UnsignedInteger16<T>)
            return V128<std::uint8_t>     {simde_mm_packus_epi16(m, Zero())};
         else if constexpr (CT::SignedInteger32<T>)
            return V128<std::int16_t>     {simde_mm_packs_epi32 (m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V128<std::uint16_t>    {simde_mm_packus_epi32(m, Zero())};
         else if constexpr (CT::SignedInteger64<T>) {
            #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
               return V128<std::int32_t>  {simde_mm_cvtepi64_epi32(m)};
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
               return V128<std::int32_t> {simde_mm_castpd_si128(ordered)};
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<T>) {
            #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
               return V128<std::uint32_t> {simde_mm_cvtepi64_epi32(m)};
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
               return V128<std::uint32_t> {simde_mm_castpd_si128(ordered)};
            #endif
         }
         else LANGULUS_ERROR("Can't unpack this type");
      }
   };

   using V128f   = V128<simde_float32>;
   using V128d   = V128<simde_float64>;

   using V128i8  = V128<std::int8_t>;
   using V128i16 = V128<std::int16_t>;
   using V128i32 = V128<std::int32_t>;
   using V128i64 = V128<std::int64_t>;

   using V128u8  = V128<std::uint8_t>;
   using V128u16 = V128<std::uint16_t>;
   using V128u32 = V128<std::uint32_t>;
   using V128u64 = V128<std::uint64_t>;
#endif
   
#if LANGULUS_SIMD(256BIT)
   /// 256bit register                                                        
   template<class>
   struct V256;

   template<>
   struct V256<simde_float32> {
      LANGULUS(TYPED) simde_float32;
      static constexpr int CTTI_SIMD_Trait = 256;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float32);

      simde__m256 m;

      LANGULUS(INLINED)
      V256(const simde__m256& v) noexcept
         : m {v} {}

      NOD() LANGULUS(INLINED)
      static V256 Zero() noexcept {
         return simde_mm256_setzero_ps();
      }
      LANGULUS(INLINED)
      operator simde__m256& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m256 const& () const noexcept {
         return m;
      }
   };

   template<>
   struct V256<simde_float64> {
      LANGULUS(TYPED) simde_float64;
      static constexpr int CTTI_SIMD_Trait = 256;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float64);

      simde__m256d m;

      LANGULUS(INLINED)
      V256(const simde__m256d& v) noexcept
         : m {v} {}

      NOD() LANGULUS(INLINED)
      static V256 Zero() noexcept {
         return simde_mm256_setzero_pd();
      }
      LANGULUS(INLINED)
      operator simde__m256d& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m256d const& () const noexcept {
         return m;
      }
   };

   template<IntElement T>
   struct V256<T> {
      LANGULUS(TYPED) T;
      static constexpr int CTTI_SIMD_Trait = 256;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(T);

      simde__m256i m;

      LANGULUS(INLINED)
      V256(const simde__m256i& v) noexcept
         : m {v} {}

      NOD() LANGULUS(INLINED)
      static V256 Zero() noexcept {
         return simde_mm256_setzero_si256();
      }
      LANGULUS(INLINED)
      operator simde__m256i& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m256i const& () const noexcept {
         return m;
      }

      NOD() LANGULUS(INLINED)
      auto UnpackLo() const noexcept {
         if constexpr (CT::SignedInteger8<T>)
            return V256<std::int16_t>  {simde_mm256_cvtepi8_epi16 (simde_mm256_extractf128_si256(m, 0))};
         else if constexpr (CT::UnsignedInteger8<T>)
            return V256<std::uint16_t> {simde_mm256_cvtepu8_epi16 (simde_mm256_extractf128_si256(m, 0))};
         else if constexpr (CT::SignedInteger16<T>)
            return V256<std::int32_t>  {simde_mm256_cvtepi16_epi32(simde_mm256_extractf128_si256(m, 0))};
         else if constexpr (CT::UnsignedInteger16<T>)
            return V256<std::uint32_t> {simde_mm256_cvtepu16_epi32(simde_mm256_extractf128_si256(m, 0))};
         else if constexpr (CT::SignedInteger32<T>)
            return V256<std::int64_t>  {simde_mm256_cvtepi32_epi64(simde_mm256_extractf128_si256(m, 0))};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V256<std::uint64_t> {simde_mm256_cvtepu32_epi64(simde_mm256_extractf128_si256(m, 0))};
         else
            LANGULUS_ERROR("Can't unpack this type");
      }

      NOD() LANGULUS(INLINED)
      auto UnpackHi() const noexcept {
         if constexpr (CT::SignedInteger8<T>)
            return V256<std::int16_t>  {simde_mm256_cvtepi8_epi16 (simde_mm256_extractf128_si256(m, 1))};
         else if constexpr (CT::UnsignedInteger8<T>)
            return V256<std::uint16_t> {simde_mm256_cvtepu8_epi16 (simde_mm256_extractf128_si256(m, 1))};
         else if constexpr (CT::SignedInteger16<T>)
            return V256<std::int32_t>  {simde_mm256_cvtepi16_epi32(simde_mm256_extractf128_si256(m, 1))};
         else if constexpr (CT::UnsignedInteger16<T>)
            return V256<std::uint32_t> {simde_mm256_cvtepu16_epi32(simde_mm256_extractf128_si256(m, 1))};
         else if constexpr (CT::SignedInteger32<T>)
            return V256<std::int64_t>  {simde_mm256_cvtepi32_epi64(simde_mm256_extractf128_si256(m, 1))};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V256<std::uint64_t> {simde_mm256_cvtepu32_epi64(simde_mm256_extractf128_si256(m, 1))};
         else
            LANGULUS_ERROR("Can't unpack this type");
      }

      NOD() LANGULUS(INLINED)
      auto Pack() const noexcept {
         if constexpr (CT::Integer8<T>)
            return *this;
         else if constexpr (CT::SignedInteger16<T>) {
            const auto lo_lane = simde_mm256_castsi256_si128(m);
            const auto hi_lane = simde_mm256_extracti128_si256(m, 1);
            return V128<std::int8_t> {simde_mm_packs_epi16(lo_lane, hi_lane)};
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            const auto lo_lane = simde_mm256_castsi256_si128(m);
            const auto hi_lane = simde_mm256_extracti128_si256(m, 1);
            return V128<std::uint8_t> {simde_mm_packus_epi16(lo_lane, hi_lane)};
         }
         else if constexpr (CT::SignedInteger32<T>)
            return V256<std::int16_t>     {simde_mm256_packs_epi32 (m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V256<std::uint16_t>    {simde_mm256_packus_epi32(m, Zero())};
         else if constexpr (CT::SignedInteger64<T>) {
            #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
               return V128<std::int32_t>  {simde_mm256_cvtepi64_epi32(m)};
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

               return V256<std::int32_t>  {simde_mm256_castpd_si256(ordered)};
            #endif
         }
         else if constexpr (CT::UnsignedInteger64<T>) {
            #if LANGULUS_SIMD(AVX512F) and LANGULUS_SIMD(AVX512VL)
               return V128<std::uint32_t> {simde_mm256_cvtepi64_epi32(m)};
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

               return V256<std::uint32_t>  {simde_mm256_castpd_si256(ordered)};
            #endif
         }
         else LANGULUS_ERROR("Can't unpack this type");
      }
   };

   using V256f   = V256<simde_float32>;
   using V256d   = V256<simde_float64>;

   using V256i8  = V256<std::int8_t>;
   using V256i16 = V256<std::int16_t>;
   using V256i32 = V256<std::int32_t>;
   using V256i64 = V256<std::int64_t>;

   using V256u8  = V256<std::uint8_t>;
   using V256u16 = V256<std::uint16_t>;
   using V256u32 = V256<std::uint32_t>;
   using V256u64 = V256<std::uint64_t>;
#endif

#if LANGULUS_SIMD(512BIT)
   /// 512bit register                                                        
   template<class>
   struct V512;

   template<>
   struct V512<simde_float32> {
      LANGULUS(TYPED) simde_float32;
      static constexpr int CTTI_SIMD_Trait = 512;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float32);

      simde__m512 m;

      LANGULUS(INLINED)
      V512(const simde__m512& v) noexcept
         : m {v} {}

      NOD() LANGULUS(INLINED)
      static V512 Zero() noexcept {
         return simde_mm512_setzero_ps();
      }
      LANGULUS(INLINED)
      operator simde__m512& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m512 const& () const noexcept {
         return m;
      }
   };

   template<>
   struct V512<simde_float64> {
      LANGULUS(TYPED) simde_float64;
      static constexpr int CTTI_SIMD_Trait = 512;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(simde_float64);

      simde__m512d m;

      LANGULUS(INLINED)
      V512(const simde__m512d& v) noexcept
         : m {v} {}

      NOD() LANGULUS(INLINED)
      static V512 Zero() noexcept {
         return simde_mm512_setzero_pd();
      }
      LANGULUS(INLINED)
      operator simde__m512d& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m512d const& () const noexcept {
         return m;
      }
   };

   template<IntElement T>
   struct V512<T> {
      LANGULUS(TYPED) T;
      static constexpr int CTTI_SIMD_Trait = 512;
      static constexpr Count MemberCount = (CTTI_SIMD_Trait / 8) / sizeof(T);

      simde__m512i m;

      LANGULUS(INLINED)
      V512(const simde__m512i& v) noexcept
         : m {v} {}

      NOD() LANGULUS(INLINED)
      static V512 Zero() noexcept {
         return simde_mm512_setzero_si512();
      }
      LANGULUS(INLINED)
      operator simde__m512i& () noexcept {
         return m;
      }
      LANGULUS(INLINED)
      operator simde__m512i const& () const noexcept {
         return m;
      }
      
      NOD() LANGULUS(INLINED)
      auto UnpackLo() const noexcept {
         if constexpr (CT::SignedInteger8<T>)
            return V512<std::int16_t>  {simde_mm512_unpacklo_epi8 (m, Zero())};
         else if constexpr (CT::UnsignedInteger8<T>)
            return V512<std::uint16_t> {simde_mm512_unpacklo_epi8 (m, Zero())};
         else if constexpr (CT::SignedInteger16<T>)
            return V512<std::int32_t>  {simde_mm512_unpacklo_epi16(m, Zero())};
         else if constexpr (CT::UnsignedInteger16<T>)
            return V512<std::uint32_t> {simde_mm512_unpacklo_epi16(m, Zero())};
         else if constexpr (CT::SignedInteger32<T>)
            return V512<std::int64_t>  {simde_mm512_unpacklo_epi32(m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V512<std::uint64_t> {simde_mm512_unpacklo_epi32(m, Zero())};
         else
            LANGULUS_ERROR("Can't unpack this type");
      }

      NOD() LANGULUS(INLINED)
      auto UnpackHi() const noexcept {
         if constexpr (CT::SignedInteger8<T>)
            return V512<std::int16_t>  {simde_mm512_unpackhi_epi8 (m, Zero())};
         else if constexpr (CT::UnsignedInteger8<T>)
            return V512<std::uint16_t> {simde_mm512_unpackhi_epi8 (m, Zero())};
         else if constexpr (CT::SignedInteger16<T>)
            return V512<std::int32_t>  {simde_mm512_unpackhi_epi16(m, Zero())};
         else if constexpr (CT::UnsignedInteger16<T>)
            return V512<std::uint32_t> {simde_mm512_unpackhi_epi16(m, Zero())};
         else if constexpr (CT::SignedInteger32<T>)
            return V512<std::int64_t>  {simde_mm512_unpackhi_epi32(m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V512<std::uint64_t> {simde_mm512_unpackhi_epi32(m, Zero())};
         else
            LANGULUS_ERROR("Can't unpack this type");
      }

      NOD() LANGULUS(INLINED)
      auto Pack() const noexcept {
         if constexpr (CT::Integer8<T>)
            return *this;
         else if constexpr (CT::SignedInteger16<T>) {
            const auto lo_lane = simde_mm512_castsi512_si256(m);
            const auto hi_lane = simde_mm512_extracti256_si512(m, 1);
            return V256<std::int8_t> {
               simde_mm256_packs_epi16(lo_lane, hi_lane)
            }.Pack();
         }
         else if constexpr (CT::UnsignedInteger16<T>) {
            const auto lo_lane = simde_mm512_castsi512_si256(m);
            const auto hi_lane = simde_mm512_extracti256_si512(m, 1);
            return V256<std::uint8_t> {
               simde_mm256_packus_epi16(lo_lane, hi_lane)
            }.Pack();
         }
         else if constexpr (CT::SignedInteger32<T>)
            return V512<std::int16_t>  {simde_mm512_packs_epi32 (m, Zero())};
         else if constexpr (CT::UnsignedInteger32<T>)
            return V512<std::uint16_t> {simde_mm512_packus_epi32(m, Zero())};
         else if constexpr (CT::SignedInteger64<T>)
            return V256<std::int32_t>  {simde_mm512_cvtepi64_epi32(m)};
         else if constexpr (CT::UnsignedInteger64<T>)
            return V256<std::uint32_t> {simde_mm512_cvtepi64_epi32(m)};
         else
            LANGULUS_ERROR("Can't unpack this type");
      }
   };

   using V512f   = V512<simde_float32>;
   using V512d   = V512<simde_float64>;

   using V512i8  = V512<std::int8_t>;
   using V512i16 = V512<std::int16_t>;
   using V512i32 = V512<std::int32_t>;
   using V512i64 = V512<std::int64_t>;

   using V512u8  = V512<std::uint8_t>;
   using V512u16 = V512<std::uint16_t>;
   using V512u32 = V512<std::uint32_t>;
   using V512u64 = V512<std::uint64_t>;
#endif

} // namespace Langulus::SIMD

namespace Langulus::CT
{

#if LANGULUS_SIMD(128BIT)
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
#else
   template<class...T>
   concept SIMD128f = false;
   template<class...T>
   concept SIMD128d = false;
   template<class...T>
   concept SIMD128i = false;
   template<class...T>
   concept SIMD128  = false;
#endif

#if LANGULUS_SIMD(256BIT)
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
#else
   template<class...T>
   concept SIMD256f = false;
   template<class...T>
   concept SIMD256d = false;
   template<class...T>
   concept SIMD256i = false;
   template<class...T>
   concept SIMD256  = false;
#endif

#if LANGULUS_SIMD(512BIT)
   /// Concept for 512bit SIMD float registers                                
   template<class...T>
   concept SIMD512f = ((Deref<T>::CTTI_SIMD_Trait == 512
       and CT::Float<TypeOf<T>>) and ...);

   /// Concept for 512bit SIMD double registers                               
   template<class...T>
   concept SIMD512d = ((Deref<T>::CTTI_SIMD_Trait == 512
       and CT::Double<TypeOf<T>>) and ...);

   /// Concept for 512bit SIMD integer/bool registers                         
   template<class...T>
   concept SIMD512i = ((Deref<T>::CTTI_SIMD_Trait == 512
       and CT::Integer<TypeOf<T>>) and ...);

   /// Concept for 512bit SIMD registers                                      
   template<class...T>
   concept SIMD512  = ((Deref<T>::CTTI_SIMD_Trait == 512) and ...);
#else
   template<class...T>
   concept SIMD512f = false;
   template<class...T>
   concept SIMD512d = false;
   template<class...T>
   concept SIMD512i = false;
   template<class...T>
   concept SIMD512  = false;
#endif

   /// Concept for SIMD registers                                             
   template<class...T>
   concept SIMD = ((SIMD128<T> or SIMD256<T> or SIMD512<T>) and ...);

   /// Anything but SIMD registers                                            
   template<class...T>
   concept NotSIMD = ((not SIMD<T>) and ...);

} // namespace Langulus::CT


namespace Langulus::SIMD
{

   /// Get the first element of an array or vector, or just the scalar        
   template<CT::NotSIMD T> NOD() LANGULUS(INLINED)
   constexpr decltype(auto) GetFirst(const T& a) noexcept {
      if constexpr (requires { a[0]; })
         return (a[0]);
      else
         return (a);
   }

   /// Get the first element of an array or vector, or just the scalar        
   template<CT::NotSIMD T> NOD() LANGULUS(INLINED)
   constexpr decltype(auto) GetFirst(T& a) noexcept {
      if constexpr (requires { a[0]; })
         return (a[0]);
      else
         return (a);
   }

   namespace Inner
   {

      template<class LHS, class RHS>
      consteval auto LosslessArray() {
         using LT = TypeOf<Desem<LHS>>;
         using RT = TypeOf<Desem<RHS>>;
         constexpr auto C = OverlapCounts<LHS, RHS>();

         if constexpr (CT::Void<LHS, RHS>) {
            // Both sides are void                                      
            return Unsupported {};
         }
         else if constexpr (CT::Void<LHS>) {
            // LHS is void, we rely only on RHS, which can be either    
            // a register, a scalar, or an array                        
            if constexpr (CT::SIMD<RHS>)
               return RHS {};
            else if constexpr (C == 1)
               return RT {};
            else
               return std::array<RT, C> {};
         }
         else if constexpr (CT::Void<RHS>) {
            // RHS is void, we rely only on LHS, which can be either    
            // a register, a scalar, or an array                        
            if constexpr (CT::SIMD<LHS>)
               return LHS {};
            else if constexpr (C == 1)
               return LT {};
            else
               return std::array<LT, C> {};
         }
         else if constexpr (CT::SIMD<LHS> and not CT::SIMD<RHS>) {
            // Both sides are known, LHS is a register, so we rely only 
            // on RHS, which can be either scalar, or an array          
            if constexpr (CountOf<RHS> == 1)
               return RT {};
            else
               return std::array<RT, CountOf<RHS>> {};
         }
         else if constexpr (not CT::SIMD<LHS> and CT::SIMD<RHS>) {
            // Both sides are known, RHS is a register, so we rely only 
            // on LHS, which can be either scalar, or an array          
            if constexpr (CountOf<LHS> == 1)
               return LT {};
            else
               return std::array<LT, CountOf<LHS>> {};
         }
         else {
            // Both sides are known, and none are registers, so pick    
            // the most lossless of the two                             
            if constexpr (C == 1)
               return Lossless<LT, RT> {};
            else
               return std::array<Lossless<LT, RT>, C> {};
         }
      }

      template<class F, class T>
      consteval auto InvocableResultInner1() noexcept {
         if constexpr (CT::Nullptr<Decay<F>>)
            return (Unsupported*) nullptr;
         else
            return (::std::invoke_result_t<F, T>*) nullptr;
      }

      template<class F, class T>
      consteval auto InvocableResultInner2() noexcept {
         if constexpr (CT::Nullptr<Decay<F>>)
            return (Unsupported*) nullptr;
         else
            return (::std::invoke_result_t<F, T, T>*) nullptr;
      }

   } // namespace Langulus::SIMD::Inner

   /// Useful tool for auto-deducing operation return type based on arguments 
   ///   @tparam LHS - left operand                                           
   ///   @tparam RHS - right operand                                          
   template<class LHS, class RHS = LHS>
   using LosslessArray = decltype(Inner::LosslessArray<LHS, RHS>());

   /// Get the return type of F(T)                                            
   template<class F, class T>
   using InvocableResult1 = Deptr<
      decltype(Inner::InvocableResultInner1<F, T>())>;

   /// Get the return type of F(T, T)                                         
   template<class F, class T>
   using InvocableResult2 = Deptr<
      decltype(Inner::InvocableResultInner2<F, T>())>;



#if LANGULUS_SIMD(128BIT)
   /// Got these from:                                                        
   /// https://stackoverflow.com/questions/41144668                           
   LANGULUS(INLINED)
   V128d uint64_to_double_full(V128u64 x) {
      auto xH = simde_mm_srli_epi64(x.m, 32);
      xH = simde_mm_or_si128(xH, simde_mm_castpd_si128(simde_mm_set1_pd(19342813113834066795298816.)));          //  2^84
      auto xL = simde_mm_blend_epi16(x.m, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
      auto f = simde_mm_sub_pd(simde_mm_castsi128_pd(xH), simde_mm_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52
      return {simde_mm_add_pd(f, simde_mm_castsi128_pd(xL))};
   }

   LANGULUS(INLINED)
   V128d int64_to_double_full(V128i64 x) {
      auto xH = simde_mm_srai_epi32(x.m, 16);
      xH = simde_mm_blend_epi16(xH, simde_mm_setzero_si128(), 0x33);
      xH = simde_mm_add_epi64(xH, simde_mm_castpd_si128(simde_mm_set1_pd(442721857769029238784.)));              //  3*2^67
      auto xL = simde_mm_blend_epi16(x.m, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)), 0x88);   //  2^52
      auto f = simde_mm_sub_pd(simde_mm_castsi128_pd(xH), simde_mm_set1_pd(442726361368656609280.));          //  3*2^67 + 2^52
      return {simde_mm_add_pd(f, simde_mm_castsi128_pd(xL))};
   }

   /// Only works for inputs in the range: [-2^51, 2^51]                      
   LANGULUS(INLINED)
   V128d int64_to_double(V128i64 x) {
      x.m = simde_mm_add_epi64(x.m, simde_mm_castpd_si128(simde_mm_set1_pd(0x0018000000000000)));
      return {simde_mm_sub_pd(simde_mm_castsi128_pd(x.m), simde_mm_set1_pd(0x0018000000000000))};
   }

   /// Only works for inputs in the range: [0, 2^52)                          
   LANGULUS(INLINED)
   V128d uint64_to_double(V128u64 x) {
      x.m = simde_mm_or_si128(x.m, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)));
      return {simde_mm_sub_pd(simde_mm_castsi128_pd(x.m), simde_mm_set1_pd(0x0010000000000000))};
   }

   /// Only works for inputs in the range: [-2^51, 2^51]                      
   LANGULUS(INLINED)
   V128i64 double_to_int64(V128d x) {
      x.m = simde_mm_add_pd(x.m, simde_mm_set1_pd(0x0018000000000000));
      return {simde_mm_sub_epi64(
         simde_mm_castpd_si128(x.m),
         simde_mm_castpd_si128(simde_mm_set1_pd(0x0018000000000000))
      )};
   }

   /// Only works for inputs in the range: [0, 2^52)                          
   LANGULUS(INLINED)
   V128u64 double_to_uint64(V128d x) {
      x.m = simde_mm_add_pd(x.m, simde_mm_set1_pd(0x0010000000000000));
      return {simde_mm_xor_si128(
         simde_mm_castpd_si128(x.m),
         simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000))
      )};
   }
#endif

   /// Shuffle eight indices                                                  
   NOD() constexpr int Shuffle(
      int&& z1, int&& y1, int&& x1, int&& w1,
      int&& z0, int&& y0, int&& x0, int&& w0
   ) noexcept {
      // 8 indices, 4 bits each                                         
      return (z1 << 28) | (y1 << 24) | (x1 << 20) | (w1 << 16)
           | (z0 << 12) | (y0 <<  8) | (x0 <<  4) |  w0;
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

#if LANGULUS_SIMD(128BIT)
   ///                                                                        
   LANGULUS(INLINED)
   V128f _mm_halfflip(const V128f what) noexcept {
      return {simde_mm_permute_ps(what.m, Shuffle(2, 3, 0, 1))};
   }

   LANGULUS(INLINED)
   V128d _mm_halfflip(const V128d what) noexcept {
      return {simde_mm_permute_pd(what.m, Shuffle(1, 0))};
   }

   template<CT::Integer T> LANGULUS(INLINED)
   V128<T> _mm_halfflip(const V128<T> what) noexcept {
      if constexpr (sizeof(T) == 4) {
         constexpr int8_t imm8 = static_cast<int8_t>(Shuffle(0, 1, 2, 3));
         return simde_mm_shuffle_epi32(what.m, imm8);
      }
      else LANGULUS_ERROR("TODO");
   }
#endif

#if LANGULUS_SIMD(256BIT)
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

#if LANGULUS_SIMD(128BIT)
   ///                                                                        
   LANGULUS(INLINED)
   int _mm_hmax_epu8(const V128u8 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epu8(vmax, simde_mm_alignr_epi8(vmax, vmax, 1)); // SSSE3 + SSE2
      vmax = simde_mm_max_epu8(vmax, simde_mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
      vmax = simde_mm_max_epu8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epu8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi8(vmax, 0); // SSE4.1
   }

   LANGULUS(INLINED)
   int _mm_hmax_epu16(const V128u16 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epu16(vmax, simde_mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
      vmax = simde_mm_max_epu16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epu16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi16(vmax, 0); // SSE2
   }

   LANGULUS(INLINED)
   int _mm_hmax_epu32(const V128u32 v) noexcept {
      auto vmax = v.m;
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
   int _mm_hmax_epi8(const V128i8 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epi8(vmax, simde_mm_alignr_epi8(vmax, vmax, 1)); // SSSE3 + SSE2
      vmax = simde_mm_max_epi8(vmax, simde_mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
      vmax = simde_mm_max_epi8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epi8(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi8(vmax, 0); // SSE4.1
   }

   LANGULUS(INLINED)
   int _mm_hmax_epi16(const V128i16 v) noexcept {
      auto vmax = v.m;
      vmax = simde_mm_max_epi16(vmax, simde_mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
      vmax = simde_mm_max_epi16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(1, 2, 3, 0))); // SSE2
      vmax = simde_mm_max_epi16(vmax, simde_mm_shuffle_epi32(vmax, Shuffle(2, 3, 0, 1))); // SSE2
      return simde_mm_extract_epi16(vmax, 0); // SSE2
   }

   LANGULUS(INLINED)
   int _mm_hmax_epi32(const V128i32 v) noexcept {
      auto vmax = v.m;
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
   simde__m128i lgls_blendv_epi32(simde__m128i a, simde__m128i b, simde__m128i mask) {
      return simde_mm_castps_si128(simde_mm_blendv_ps(
         simde_mm_castsi128_ps(a),
         simde_mm_castsi128_ps(b),
         simde_mm_castsi128_ps(mask)
      ));
   }
#endif

#if LANGULUS_SIMD(256BIT)
   LANGULUS(INLINED)
   simde__m256i lgls_blendv_epi32(simde__m256i a, simde__m256i b, simde__m256i mask) {
      return simde_mm256_castps_si256(simde_mm256_blendv_ps(
         simde_mm256_castsi256_ps(a),
         simde_mm256_castsi256_ps(b),
         simde_mm256_castsi256_ps(mask)
      ));
   }
#endif

#if LANGULUS_SIMD(128BIT)
   /// Pack 16bit integers (signed or not) to 8bit integers with truncation   
   ///   @param low - lower eight 16bit integers                              
   ///   @param high - higher eight 16bit integers                            
   ///   @return the combined 16 truncated 8bit equivalents                   
   template<CT::Integer16 T> LANGULUS(INLINED)
   auto lgls_pack_epi16(V128<T> low, V128<T> high) {
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
#endif

#if LANGULUS_SIMD(256BIT)
   /// Pack 16bit integers (signed or not) to 8bit integers with truncation   
   ///   @param low - lower sixteen 16bit integers                            
   ///   @param high - higher sixteen 16bit integers                          
   ///   @return the combined 32 truncated 8bit equivalents                   
   template<CT::Integer16 T> LANGULUS(INLINED)
   auto lgls_pack_epi16(V256<T> low, V256<T> high) {
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
#endif

   /*inline simde__m512i lgls_blendv_epi32(const simde__m512i& a, const simde__m512i& b, const simde__m512i& mask) {
      use _mm512_mask_blend_ instead
      return simde_mm512_castps_si512(simde_mm512_blendv_ps(
         simde_mm512_castsi512_ps(a),
         simde_mm512_castsi512_ps(b),
         simde_mm512_castsi512_ps(mask)
      ));
   }*/

#if LANGULUS_SIMD(128BIT)
   /// Pack 32bit integers (signed or not) to 16bit integers with truncation  
   ///   @param low - lower four 32bit integers                               
   ///   @param high - higher four 32bit integers                             
   ///   @return the combined 8 truncated 16bit equivalents                   
   template<CT::Integer32 T> LANGULUS(INLINED)
   auto lgls_pack_epi32(V128<T> low, V128<T> high) {
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
            simde_mm_shuffle_epi8(low,  maskLo),
            simde_mm_shuffle_epi8(high, maskHi)
         );
      #endif

      if constexpr (CT::Signed<T>)
         return V128i16 {r};
      else
         return V128u16 {r};
   }
#endif

#if LANGULUS_SIMD(256BIT)
   /// Pack 32bit integers (signed or not) to 16bit integers with truncation  
   ///   @param low - lower eight 32bit integers                              
   ///   @param high - higher eight 32bit integers                            
   ///   @return the combined 16 truncated 16bit equivalents                  
   template<CT::Integer32 T> LANGULUS(INLINED)
   auto lgls_pack_epi32(V256<T> low, V256<T> high) {
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

   /// Pack 64bit integers (signed or not) to 32bit integers with truncation  
   /// https://stackoverflow.com/questions/69408063                           
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
#endif

   /*inline simde__m512i lgls_pack_epi32(const simde__m512i& a, const simde__m512i& b, const simde__m512i& mask) {
      use _mm512_mask_blend_ instead
      return simde_mm512_castps_si512(simde_mm512_blendv_ps(
         simde_mm512_castsi512_ps(a),
         simde_mm512_castsi512_ps(b),
         simde_mm512_castsi512_ps(mask)
      ));
   }*/

} // namespace Langulus::SIMD

