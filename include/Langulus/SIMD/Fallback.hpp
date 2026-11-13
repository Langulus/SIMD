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
   template<class OUT, class VAL, class FFALL> LANGULUS(INLINED)
   constexpr auto FallbackUnary(VAL& val, FFALL&& op) {
      if constexpr (CT::SIMD<VAL>) {
         // Fallback routine can't handle registers, but the function   
         // instantiation is still needed in the Evaluate function      
         return No {};
      }
      else {
         using RETURN = SIMD::LosslessArray<VAL, VAL>;
         using LOSSLESS = TypeOf<RETURN>;
         constexpr auto S = ExtentOf<RETURN>;

         if constexpr (CT::Vector<VAL>) {
            // Vector OP                                                
            RETURN output;
            for (size_t i = 0; i < S; ++i) {
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
   
   /// Fallback OP with two arguments.                                        
   /// It converts LHS and RHS to the most lossless of the two.               
   ///   @tparam OUT the desired output array/vector/scalar                   
   ///   @param op the fallback lambda to invoke                              
   ///   @return the resulting number/bool/std::array of numbers/bools, or No 
   ///      if operation is not supported for the chosen arguments            
   template<class OUT, class LHS, class RHS, class FFALL> LANGULUS(INLINED)
   constexpr auto FallbackBinary(LHS const& lhs, RHS const& rhs, FFALL&& op) {
      /*if constexpr (CT::SIMD<LHS> or CT::SIMD<RHS>) {
         // Fallback routine can't handle registers, but the function   
         // instantiation is still needed in the Evaluate function      
         return No {};
      }
      else {*/
         static_assert(    CT::NotIntent<OUT>,     "Strip intents");
         static_assert(    CT::NotReference<OUT>,  "Strip references");
         //static_assert(    CT::Array<OUT>,         "Output must be an array");
         //static_assert(not CT::BoundedArray<OUT>,  "You were supposed to wrap it in an ::std::array");

         //using RETURN = SIMD::LosslessArray<LHS, RHS>;
         //using LOSSLESS = TypeOf<RETURN>;
         //constexpr auto S = ExtentOf<RETURN>;
         using LOSSLESS = Tif<CT::Array<OUT>, TypeOf<OUT>, OUT>;
         constexpr size_t S = ExtentOf<OUT>;
         using OUT_NONBOUND = Tif<CT::BoundedArray<OUT>, ::std::array<LOSSLESS, S>, OUT>;

         if constexpr (CT::Vector<LHS, RHS>) {
            // Vector OP Vector                                         
            OUT_NONBOUND output;
            for (size_t i = 0; i < S; ++i) {
               output[i] = static_cast<LOSSLESS>(op(
                  static_cast<LOSSLESS>(lhs[i]),
                  static_cast<LOSSLESS>(rhs[i])
               ));
            }

            return output;
         }
         else if constexpr (CT::Vector<LHS>) {
            // Vector OP Scalar                                         
            OUT_NONBOUND output;
            const auto same_rhs = static_cast<LOSSLESS>(GetFirst(rhs));
            for (size_t i = 0; i < S; ++i) {
               output[i] = static_cast<LOSSLESS>(op(
                  static_cast<LOSSLESS>(lhs[i]),
                  same_rhs
               ));
            }

            return output;
         }
         else if constexpr (CT::Vector<RHS>) {
            // Scalar OP Vector                                         
            OUT_NONBOUND output;
            const auto same_lhs = static_cast<LOSSLESS>(GetFirst(lhs));
            for (size_t i = 0; i < S; ++i) {
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
      //}
   }
}
