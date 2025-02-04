///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include <Langulus/RTTI/Meta.hpp>
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

#if 0
   #define LANGULUS_SIMD_VERBOSE(...)     Logger::Info(__VA_ARGS__)
   #define LANGULUS_SIMD_VERBOSE_TAB(...) const auto scoped = Logger::InfoTab(__VA_ARGS__)
#else
   #define LANGULUS_SIMD_VERBOSE(...)     LANGULUS(NOOP)
   #define LANGULUS_SIMD_VERBOSE_TAB(...) LANGULUS(NOOP)
#endif


///                                                                           
///   Detect available SIMD                                                   
///                                                                           
/// By default nothing is enabled                                             
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
      signed int, signed long, unsigned int, unsigned long,
      ::std::int8_t,  ::std::int16_t,  ::std::int32_t,  ::std::int64_t,
      ::std::uint8_t, ::std::uint16_t, ::std::uint32_t, ::std::uint64_t,
      char8_t, char16_t, char32_t, wchar_t, Langulus::Byte
   >) and ...);

   /// Single element inside a register                                       
   template<class...T>
   concept Element = RealElement<T...> or IntElement<T...>;


#if LANGULUS_SIMD(512BIT)
   /// 512bit register                                                        
   template<class>
   struct V512;

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

#if LANGULUS_SIMD(256BIT)
   /// 256bit register                                                        
   template<class>
   struct V256;

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

#if LANGULUS_SIMD(128BIT)
   /// 128bit register                                                        
   template<class>
   struct V128;

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
   

   ///                                                                        
   /// The following functions generate shuffle masks, often called immX      
   /// paremeters in intrinsics. You should be careful how much bits each     
   /// index takes, best found out by reading the operation code for the      
   /// intrinsic it's applied to.                                             
   ///                                                                        
   /// Here's an example with _mm256_permute2f128_ps:                         
   /// DEFINE SELECT4(src1, src2, control) {                                  
   ///    CASE(control[1:0]) OF                        // uses control[1:0] - 
   ///       0:  tmp[127:0] : = src1[127:0]            // that is two bits,   
   ///       1:  tmp[127:0] : = src1[255:128]          // and checks value in 
   ///       2:  tmp[127:0] : = src2[127:0]            // those bits [0;3]    
   ///       3:  tmp[127:0] : = src2[255:128]                                 
   ///    ESAC                                                                
   ///    IF control[3]                                // however it also uses
   ///       tmp[127:0] : = 0                          // fourth bit for      
   ///    FI                                           // zeroing             
   ///    RETURN tmp[127:0]                                                   
   /// }                                                                      
   /// dst[127:0]   : = SELECT4(a[255:0], b[255:0], imm8[3:0])  // so overall, the operation takes          
   /// dst[255:128] : = SELECT4(a[255:0], b[255:0], imm8[7:4])  // 4 bits for two indices, thus we should   
   /// dst[MAX:256] : = 0                                       // use Shuffle4(x,y) to set that up         
   ///                                                                                                      
   /// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permute2f128_ps 
   /// Other functions like _mm_shuffle_epi32 often use Shuffle2(x,y,z,w)     
   /// The compiler should complain if a shuffle masks goes beyond the valid  
   /// range for the specific intrinsic.                                      
   ///                                                                        

   /// Shuffle configuration with up to eight indices, 4 bits each            
   consteval int Shuffle4(
      int a0    , int a1 = 0, int a2 = 0, int a3 = 0,
      int a4 = 0, int a5 = 0, int a6 = 0, int a7 = 0
   ) {
      return (a7 << 28) | (a6 << 24) | (a5 << 20) | (a4 << 16)
           | (a3 << 12) | (a2 <<  8) | (a1 <<  4) |  a0;
   }

   /// Shuffle configuration with up to four indices, 2 bits each             
   consteval int Shuffle2(int a0, int a1, int a2, int a3) {
      return (a3 << 6) | (a2 << 4) | (a1 << 2) | a0;
   }

   /// Shuffle configuration with up to two indices, 1 bit each               
   consteval int Shuffle1(int a0, int a1) {
      return (a1 << 1) | a0;
   }

} // namespace Langulus::SIMD


/// Include the register types, which are designed to be seamlessly           
/// interchangable with the intrinsic types with zero overhead                
#if LANGULUS_SIMD(512BIT)
   #include "registers/V512.hpp"
#endif

#if LANGULUS_SIMD(256BIT)
   #include "registers/V256.hpp"
#endif

#if LANGULUS_SIMD(128BIT)
   #include "registers/V128.hpp"
#endif


/// Add some SIMD related concepts to the CT library                          
namespace Langulus::CT
{

#if not LANGULUS_SIMD(128BIT)
   template<class...T> concept SIMD128f = false;
   template<class...T> concept SIMD128d = false;
   template<class...T> concept SIMD128i = false;
   template<class...T> concept SIMD128  = false;
#endif

#if not LANGULUS_SIMD(256BIT)
   template<class...T> concept SIMD256f = false;
   template<class...T> concept SIMD256d = false;
   template<class...T> concept SIMD256i = false;
   template<class...T> concept SIMD256  = false;
#endif

#if not LANGULUS_SIMD(512BIT)
   template<class...T> concept SIMD512f = false;
   template<class...T> concept SIMD512d = false;
   template<class...T> concept SIMD512i = false;
   template<class...T> concept SIMD512  = false;
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
   LANGULUS(INLINED)
   constexpr decltype(auto) GetFirst(const CT::NotSIMD auto& a) noexcept {
      if constexpr (requires { a[0]; })
         return (a[0]);
      else
         return (a);
   }

   /// Get the first element of an array or vector, or just the scalar        
   LANGULUS(INLINED)
   constexpr decltype(auto) GetFirst(CT::NotSIMD auto& a) noexcept {
      if constexpr (requires { a[0]; })
         return (a[0]);
      else
         return (a);
   }

   namespace Inner
   {

      template<class T>
      consteval auto LosslessRegister() {
         #if LANGULUS_SIMD(128BIT)
            if constexpr (sizeof(Deint<T>) <= 16)
               return (V128<TypeOf<Deint<T>>>*) nullptr;
            else
         #endif
         #if LANGULUS_SIMD(256BIT)
            if constexpr (sizeof(Deint<T>) <= 32)
               return (V256<TypeOf<Deint<T>>>*) nullptr;
            else
         #endif
         #if LANGULUS_SIMD(512BIT)
            if constexpr (sizeof(Deint<T>) <= 64)
               return (V512<TypeOf<Deint<T>>>*) nullptr;
            else
         #endif
            static_assert(false, "Unsupported register");
      }

      template<class LHS, class RHS>
      consteval auto LosslessArray() {
         using LT = TypeOf<Deint<LHS>>;
         using RT = TypeOf<Deint<RHS>>;
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

   template<class T>
   using LosslessRegister = Deptr<decltype(Inner::LosslessRegister<T>())>;

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

   /// Clamp a real value inside the interval [0:1]                           
   /// Clamp integers in the numerical limits of provided AS                  
   ///   @param v - saturate the value by converting T to a smaller type and  
   ///      clamping to the min/max if value exceeds the smaller range        
   template<class AS, class T> LANGULUS(INLINED)
   constexpr AS Saturate(const T& v) noexcept {
      if constexpr (CT::Real<T>)
         return static_cast<AS>(v < T {0} ? T {0} : v > T {1} ? T {1} : v);
      else if constexpr (CT::Integer<T> and sizeof(AS) < sizeof(T)) {
         constexpr T low = static_cast<T>(::std::numeric_limits<AS>::min());
         constexpr T hi  = static_cast<T>(::std::numeric_limits<AS>::max());
         return static_cast<AS>(v > hi ? hi : (v < low ? low : v));
      }
      else return static_cast<AS>(v);
   }

} // namespace Langulus::SIMD

namespace Langulus::CT
{
   /// Anything that is saturated                                             
   /// Notice that only one of the types has to be saturated                  
   template<class...T>
   concept Saturated = ((Decay<Deint<T>>::CTTI_SaturatedTrait) or ...);

   template<class...T>
   concept Unsaturated = ((not Saturated<T>) and ...);
}