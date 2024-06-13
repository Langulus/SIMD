///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "Bitmask.hpp"


namespace Langulus::SIMD::Inner
{

   /// Fallback OP with one argument                                          
   ///   @tparam OUT - the desired output array/vector/scalar                 
   ///   @param val - argument                                                
   ///   @param op - the fallback function to invoke                          
   ///   @return the resulting number/bool/std::array of numbers/bools        
   template<class OUT, class VAL, class FFALL>
   NOD() LANGULUS(INLINED)
   constexpr auto FallbackUnary(VAL& val, FFALL&& op) {
      if constexpr (CT::SIMD<VAL>) {
         // Fallback routine can't handle registers, but the function   
         // instantiation is still needed in the Evaluate function      
         return Unsupported {};
      }
      else {
         using RETURN = SIMD::LosslessArray<VAL, VAL>;
         using LOSSLESS = TypeOf<RETURN>;
         constexpr auto S = CountOf<RETURN>;

         if constexpr (CT::Vector<VAL>) {
            // Vector OP                                                
            RETURN output;
            for (Count i = 0; i < S; ++i) {
               output[i] = static_cast<LOSSLESS>(op(
                  static_cast<LOSSLESS>(val[i])
               ));
            }

            return output;
         }
         else {
            // Scalar OP                                                
            // Casts are no-op if types are the same                    
            return static_cast<LOSSLESS>(op(
               static_cast<LOSSLESS>(GetFirst(val))
            ));
         }
      }
   }
   
   /// Fallback OP with two arguments                                         
   /// It converts LHS and RHS to the most lossless of the two                
   ///   @tparam OUT - the desired output array/vector/scalar                 
   ///   @param lhs - left argument                                           
   ///   @param rhs - right argument                                          
   ///   @param op - the fallback function to invoke                          
   ///   @return the resulting number/bool/std::array of numbers/bools        
   template<class OUT, class LHS, class RHS, class FFALL>
   NOD() LANGULUS(INLINED)
   constexpr auto FallbackBinary(LHS& lhs, RHS& rhs, FFALL&& op) {
      if constexpr (CT::SIMD<LHS> or CT::SIMD<RHS>) {
         // Fallback routine can't handle registers, but the function   
         // instantiation is still needed in the Evaluate function      
         return Unsupported {};
      }
      else {
         using RETURN = SIMD::LosslessArray<LHS, RHS>;
         using LOSSLESS = TypeOf<RETURN>;
         constexpr auto S = CountOf<RETURN>;

         if constexpr (CT::Vector<LHS, RHS>) {
            // Vector OP Vector                                         
            RETURN output;
            for (Count i = 0; i < S; ++i) {
               output[i] = static_cast<LOSSLESS>(op(
                  static_cast<LOSSLESS>(lhs[i]),
                  static_cast<LOSSLESS>(rhs[i])
               ));
            }

            return output;
         }
         else if constexpr (CT::Vector<LHS>) {
            // Vector OP Scalar                                         
            RETURN output;
            const auto same_rhs = static_cast<LOSSLESS>(GetFirst(rhs));
            for (Count i = 0; i < S; ++i) {
               output[i] = static_cast<LOSSLESS>(op(
                  static_cast<LOSSLESS>(lhs[i]),
                  same_rhs
               ));
            }

            return output;
         }
         else if constexpr (CT::Vector<RHS>) {
            // Scalar OP Vector                                         
            RETURN output;
            const auto same_lhs = static_cast<LOSSLESS>(GetFirst(lhs));
            for (Count i = 0; i < S; ++i) {
               output[i] = static_cast<LOSSLESS>(op(
                  same_lhs,
                  static_cast<LOSSLESS>(rhs[i])
               ));
            }

            return output;
         }
         else {
            // Scalar OP Scalar                                         
            // Casts are no-op if types are the same                    
            return static_cast<LOSSLESS>(op(
               static_cast<LOSSLESS>(GetFirst(lhs)),
               static_cast<LOSSLESS>(GetFirst(rhs))
            ));
         }
      }
   }

} // namespace Langulus::SIMD::Inner
