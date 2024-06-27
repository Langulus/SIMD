///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "Fallback.hpp"
#include "Convert.hpp"


namespace Langulus::SIMD::Inner
{

   /// Attempt register encapsulation of argument                             
   /// Check if result of opSIMD is supported and return it, otherwise        
   /// fallback to opFALL and calculate conventionally (can be constexpr)     
   ///   @tparam DEF - default value to fill empty register regions           
   ///      useful to avoid division-by-zero cases                            
   ///   @tparam FORCE_OUT - the type of data we want as a result - use void  
   ///      to pick a lossless type derived from 'lhs' and 'rhs'              
   ///   @param val - argument                                                
   ///   @param opSIMD - the SIMD function to invoke if supported             
   ///   @param opFALL - the fallback (non-SIMD/constexpr) function           
   ///   @return the result - either scalar, vector or register               
   template<auto DEF, class FORCE_OUT = void> NOD() LANGULUS(INLINED)
   constexpr auto AttemptUnary(
      const auto& val,
      const auto& opSIMD,
      const auto& opFALL
   ) {
      using VAL = Deref<decltype(val)>;
      using OUT = Conditional<CT::Void<FORCE_OUT>,
         SIMD::LosslessArray<VAL>,
         SIMD::LosslessArray<FORCE_OUT>
      >;
      using E = TypeOf<OUT>;
      using R = decltype(Load<DEF>(Fake<const SIMD::LosslessArray<VAL>&>()));
      constexpr bool supported = CT::SIMD<InvocableResult1<decltype(opSIMD), R>>;

      if constexpr (not supported) {
         // Operating on scalars, or SIMD not supported, just fallback  
         return FallbackUnary<OUT>(val, opFALL);
      }
      else if constexpr (not CT::SIMD<decltype(Load<DEF>(val))>) {
         // Arguments can't be loaded in registers, just fallback       
         return FallbackUnary<OUT>(val, opFALL);
      }
      else if constexpr (CT::Bool<E>) {
         // If FORCE_OUT was boolean, we're doing some comparing, so    
         // don't convert to output data yet                            
         return opSIMD(Load<DEF>(val));
      }
      else {
         // Load argument, convert it to the desired FORCE_OUT          
         // and perform the operation                                   
         const CT::SIMD auto load = Load<DEF>(val);

         if constexpr (not CT::SIMD<decltype(ConvertSIMD<E>(load))>) {
            // Arguments can't be converted to the desired type         
            return FallbackUnary<OUT>(val, opFALL);
         }
         else {
            // Perform the SIMD operation                               
            return opSIMD(ConvertSIMD<E>(load));
         }
      }
   }

   /// Attempt register encapsulation of arguments                            
   /// Check if result of opSIMD is supported and return it, otherwise        
   /// fallback to opFALL and calculate conventionally (can be constexpr)     
   ///   @tparam DEF - default value to fill empty register regions           
   ///      useful to avoid division-by-zero cases                            
   ///   @tparam FORCE_OUT - the type of data we want as a result - use void  
   ///      to pick a lossless type derived from 'lhs' and 'rhs'              
   ///   @param lhs - left argument                                           
   ///   @param rhs - right argument                                          
   ///   @param opSIMD - the SIMD function to invoke if supported             
   ///   @param opFALL - the fallback (non-SIMD/constexpr) function           
   ///   @return the result - either scalar, vector or register               
   template<auto DEF, class FORCE_OUT = void> NOD() LANGULUS(INLINED)
   constexpr auto AttemptBinary(
      const CT::NotSemantic auto& lhs,
      const CT::NotSemantic auto& rhs,
      const auto& opSIMD,
      const auto& opFALL
   ) {
      using LHS = Deref<decltype(lhs)>;
      using RHS = Deref<decltype(rhs)>;
      using LOSSLESS = SIMD::LosslessArray<LHS, RHS>;
      using OUT = Conditional<CT::Void<FORCE_OUT>,
         LOSSLESS, SIMD::LosslessArray<FORCE_OUT>>;
      using E = TypeOf<OUT>;
      using R = decltype(Load<DEF>(Fake<const LOSSLESS&>()));
      constexpr bool supported = CT::SIMD<InvocableResult2<decltype(opSIMD), R>>;

      if constexpr (not supported) {
         // Operating on scalars, or SIMD not supported, just fallback  
         return FallbackBinary<OUT>(lhs, rhs, opFALL);
      }
      else if constexpr (not CT::SIMD<decltype(Load<DEF, R>(lhs))>
                      or not CT::SIMD<decltype(Load<DEF, R>(rhs))>) {
         // Arguments can't be loaded in registers, just fallback       
         return FallbackBinary<OUT>(lhs, rhs, opFALL);
      }
      else if constexpr (CT::Bool<E>) {
         // If FORCE_OUT was boolean, we're doing some comparing, so    
         // don't convert to output data yet. Instead, convert to the   
         // lossless of the two types.                                  
         const CT::SIMD auto loadL = Load<DEF, R>(lhs);
         const CT::SIMD auto loadR = Load<DEF, R>(rhs);
         using ALT_E = TypeOf<LOSSLESS>;

         if constexpr (not CT::SIMD<decltype(ConvertSIMD<ALT_E>(loadL))>
                    or not CT::SIMD<decltype(ConvertSIMD<ALT_E>(loadR))>) {
            // Arguments can't be converted to the desired type         
            return FallbackBinary<OUT>(lhs, rhs, opFALL);
         }
         else {
            // Perform the SIMD operation                               
            return opSIMD(ConvertSIMD<ALT_E>(loadL), ConvertSIMD<ALT_E>(loadR));
         }
      }
      else {
         // Load both arguments, convert them to the desired FORCE_OUT  
         // and perform the operation                                   
         const CT::SIMD auto loadL = Load<DEF, R>(lhs);
         const CT::SIMD auto loadR = Load<DEF, R>(rhs);

         if constexpr (not CT::SIMD<decltype(ConvertSIMD<E>(loadL))>
                    or not CT::SIMD<decltype(ConvertSIMD<E>(loadR))>) {
            // Arguments can't be converted to the desired type         
            return FallbackBinary<OUT>(lhs, rhs, opFALL);
         }
         else {
            // Perform the SIMD operation                               
            return opSIMD(ConvertSIMD<E>(loadL), ConvertSIMD<E>(loadR));
         }
      }
   }

} // namespace Langulus::SIMD::Inner
