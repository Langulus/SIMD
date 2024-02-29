///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#pragma once
#include "Fill.hpp"
#include "Convert.hpp"


namespace Langulus::SIMD
{

   enum class LogStyle {
      Natural,
      Base10,
      Base1P,
      Base2,
      FlooredBase2
   };

   namespace Inner
   {

      /// Get natural/base-10/1p/base-2/floor(log2(x)) logarithm values       
      ///   @tparam STYLE - the type of the log function                      
      ///   @tparam T - the type of the array element                         
      ///   @tparam REGISTER - the register type (deducible)                  
      ///   @param value - the array                                          
      ///   @return the logarithm values                                      
      template<LogStyle STYLE = LogStyle::Base10, CT::Decayed T, CT::SIMD REGISTER>
      LANGULUS(INLINED)
      REGISTER Log(UNUSED() const REGISTER& value) noexcept {
         static_assert(CT::Real<T>, "Doesn't work for whole numbers");

      #if LANGULUS_SIMD(128BIT)
         if constexpr (CT::SIMD128<REGISTER>) {
            if constexpr (CT::Float<T>) {
               if constexpr (STYLE == LogStyle::Natural)
                  return simde_mm_log_ps(value);
               else if constexpr (STYLE == LogStyle::Base10)
                  return simde_mm_log10_ps(value);
               else if constexpr (STYLE == LogStyle::Base1P)
                  return simde_mm_log1p_ps(value);
               else if constexpr (STYLE == LogStyle::Base2)
                  return simde_mm_log2_ps(value);
               else if constexpr (STYLE == LogStyle::FlooredBase2)
                  return simde_mm_logb_ps(value);
               else LANGULUS_ERROR("Unsupported style for float[4] package");
            }
            else if constexpr (CT::Double<T>) {
               if constexpr (STYLE == LogStyle::Natural)
                  return simde_mm_log_pd(value);
               else if constexpr (STYLE == LogStyle::Base10)
                  return simde_mm_log10_pd(value);
               else if constexpr (STYLE == LogStyle::Base1P)
                  return simde_mm_log1p_pd(value);
               else if constexpr (STYLE == LogStyle::Base2)
                  return simde_mm_log2_pd(value);
               else if constexpr (STYLE == LogStyle::FlooredBase2)
                  return simde_mm_logb_pd(value);
               else LANGULUS_ERROR("Unsupported style for double[2] package");
            }
            else LANGULUS_ERROR("Unsupported type for 16-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(256BIT)
         if constexpr (CT::SIMD256<REGISTER>) {
            if constexpr (CT::Float<T>) {
               if constexpr (STYLE == LogStyle::Natural)
                  return simde_mm256_log_ps(value);
               else if constexpr (STYLE == LogStyle::Base10)
                  return simde_mm256_log10_ps(value);
               else if constexpr (STYLE == LogStyle::Base1P)
                  return simde_mm256_log1p_ps(value);
               else if constexpr (STYLE == LogStyle::Base2)
                  return simde_mm256_log2_ps(value);
               else if constexpr (STYLE == LogStyle::FlooredBase2)
                  return simde_mm256_logb_ps(value);
               else LANGULUS_ERROR("Unsupported style for float[8] package");
            }
            else if constexpr (CT::Double<T>) {
               if constexpr (STYLE == LogStyle::Natural)
                  return simde_mm256_log_pd(value);
               else if constexpr (STYLE == LogStyle::Base10)
                  return simde_mm256_log10_pd(value);
               else if constexpr (STYLE == LogStyle::Base1P)
                  return simde_mm256_log1p_pd(value);
               else if constexpr (STYLE == LogStyle::Base2)
                  return simde_mm256_log2_pd(value);
               else if constexpr (STYLE == LogStyle::FlooredBase2)
                  return simde_mm256_logb_pd(value);
               else LANGULUS_ERROR("Unsupported style for double[4] package");
            }
            else LANGULUS_ERROR("Unsupported type for 32-byte package");
         }
         else
      #endif

      #if LANGULUS_SIMD(512BIT)
         if constexpr (CT::SIMD512<REGISTER>) {
            if constexpr (CT::Float<T>) {
               if constexpr (STYLE == LogStyle::Natural)
                  return simde_mm512_log_ps(value);
               else if constexpr (STYLE == LogStyle::Base10)
                  return simde_mm512_log10_ps(value);
               else if constexpr (STYLE == LogStyle::Base1P)
                  return simde_mm512_log1p_ps(value);
               else if constexpr (STYLE == LogStyle::Base2)
                  return simde_mm512_log2_ps(value);
               else if constexpr (STYLE == LogStyle::FlooredBase2)
                  return simde_mm512_logb_ps(value);
               else LANGULUS_ERROR("Unsupported style for float[16] package");
            }
            else if constexpr (CT::Double<T>) {
               if constexpr (STYLE == LogStyle::Natural)
                  return simde_mm512_log_pd(value);
               else if constexpr (STYLE == LogStyle::Base10)
                  return simde_mm512_log10_pd(value);
               else if constexpr (STYLE == LogStyle::Base1P)
                  return simde_mm512_log1p_pd(value);
               else if constexpr (STYLE == LogStyle::Base2)
                  return simde_mm512_log2_pd(value);
               else if constexpr (STYLE == LogStyle::FlooredBase2)
                  return simde_mm512_logb_pd(value);
               else LANGULUS_ERROR("Unsupported style for double[8] package");
            }
            else LANGULUS_ERROR("Unsupported type for 64-byte package");
         }
         else
      #endif
            LANGULUS_ERROR("Unsupported type");
      }

   } // namespace Langulus::SIMD::Inner


   /// Get the logarithm values                                               
   ///   @param STYLE - what flavour of logarithm are we doing here           
   ///   @param T - type of a single value                                    
   ///   @return a register, if viable SIMD routine exists                    
   ///           or array/scalar if no viable SIMD routine exists             
   template<LogStyle STYLE = LogStyle::Base10, class T> LANGULUS(INLINED)
   auto Log(const T& value) noexcept {
      using DT = Decay<TypeOf<T>>;
      return Inner::Log<STYLE, DT>(Load<0>(value));
   }

} // namespace Langulus::SIMD