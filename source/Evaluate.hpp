///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// SPDX-License-Identifier: MIT                                              
///                                                                           
#pragma once
#include "Convert.hpp"
#include "Bitmask.hpp"


namespace Langulus::SIMD::Inner
{

   /// Fallback OP with one argument                                          
   ///   @tparam OUT - the desired output array/vector/scalar                 
   ///   @tparam VAL - array/vector/scalar  (deducible)                       
   ///   @tparam FFALL - the operation to invoke on fallback (deducible)      
   ///   @param val - argument                                                
   ///   @param op - the fallback function to invoke                          
   ///   @return the resulting number/bool/std::array of numbers/bools        
   template<class OUT, class VAL, class FFALL>
   NOD() LANGULUS(INLINED)
   constexpr auto Fallback1(VAL& val, FFALL&& op) {
      using RETURN = SIMD::LosslessArray<VAL, VAL>;
      using LOSSLESS = TypeOf<RETURN>;
      constexpr auto S = CountOf<RETURN>;

      if constexpr (CT::Vector<VAL>) {
         // Vector OP                                                   
         RETURN output;
         for (Count i = 0; i < S; ++i) {
            output[i] = static_cast<LOSSLESS>(op(
               static_cast<LOSSLESS>(DenseCast(val[i]))
            ));
         }

         return output;
      }
      else {
         // Scalar OP                                                   
         // Casts are no-op if types are the same                       
         return static_cast<LOSSLESS>(op(
            static_cast<LOSSLESS>(DenseCast(GetFirst(val)))
         ));
      }
   }

   /// Attempt register encapsulation of LHS and RHS arrays                   
   /// Check if result of opSIMD is supported and return it, otherwise        
   /// fallback to opFALL and calculate conventionally                        
   ///   @tparam DEF - default value to fill empty register regions           
   ///                 useful against division-by-zero cases                  
   ///   @tparam REGISTER - the register to use for the SIMD operation        
   ///   @tparam OUT - the type of data we want as a result                   
   ///   @tparam VAL - number type (deducible)                                
   ///   @tparam FSIMD - the SIMD operation to invoke (deducible)             
   ///   @tparam FFALL - the fallback operation to invoke (deducible)         
   ///   @param val - argument                                                
   ///   @param opSIMD - the function to invoke                               
   ///   @param opFALL - the function to invoke                               
   ///   @return the result (either std::array, number, or register)          
   template<auto DEF, class REGISTER, class OUT, class VAL, class FSIMD, class FFALL>
   NOD() LANGULUS(INLINED)
   constexpr auto Evaluate1(const VAL& val, FSIMD&& opSIMD, FFALL&& opFALL) {
      using OUTSIMD = InvocableResult1<FSIMD, REGISTER>;
      constexpr auto S = CountOf<VAL>;
      LANGULUS_SIMD_VERBOSE_TAB("Evaluated to count of ", S);

      if constexpr (S < 2 or CT::NotSIMD<REGISTER> or CT::NotSIMD<OUTSIMD>) {
         // Call the fallback routine if unsupported, or size 1         
         return Fallback1<OUT>(val, ::std::move(opFALL));
      }
      else if constexpr (CT::Vector<VAL>) {
         // VAL is vector, so wrap in a register                        
         LANGULUS_SIMD_VERBOSE("Both sides are vectors");
         return opSIMD(Convert<DEF, OUT>(val));
      }
      else {
         // VAL is scalar (or anything else)                            
         // Just fallback and use the appropriate operator              
         return Fallback1<OUT>(val, ::std::move(opFALL));
      }
   }

   
   /// Fallback OP on a single pair of dense numbers                          
   /// It converts LHS and RHS to the most lossless of the two                
   ///   @tparam OUT - the desired output array/vector/scalar                 
   ///   @tparam LHS - left array/vector/scalar  (deducible)                  
   ///   @tparam RHS - right array/vector/scalar  (deducible)                 
   ///   @tparam FFALL - the operation to invoke on fallback (deducible)      
   ///   @param lhs - left argument                                           
   ///   @param rhs - right argument                                          
   ///   @param op - the fallback function to invoke                          
   ///   @return the resulting number/bool/std::array of numbers/bools        
   template<class OUT, class LHS, class RHS, class FFALL>
   NOD() LANGULUS(INLINED)
   constexpr auto Fallback2(LHS& lhs, RHS& rhs, FFALL&& op) {
      using RETURN = SIMD::LosslessArray<LHS, RHS>;
      using LOSSLESS = TypeOf<RETURN>;
      constexpr auto S = CountOf<RETURN>;

      if constexpr (CT::Vector<LHS> and CT::Vector<RHS>) {
         // Vector OP Vector                                            
         RETURN output;
         for (Count i = 0; i < S; ++i) {
            output[i] = static_cast<LOSSLESS>(op(
               static_cast<LOSSLESS>(DenseCast(lhs[i])),
               static_cast<LOSSLESS>(DenseCast(rhs[i]))
            ));
         }

         return output;
      }
      else if constexpr (CT::Vector<LHS>) {
         // Vector OP Scalar                                            
         RETURN output;
         const auto same_rhs = static_cast<LOSSLESS>(DenseCast(GetFirst(rhs)));
         for (Count i = 0; i < S; ++i) {
            output[i] = static_cast<LOSSLESS>(op(
               static_cast<LOSSLESS>(DenseCast(lhs[i])),
               same_rhs
            ));
         }

         return output;
      }
      else if constexpr (CT::Vector<RHS>) {
         // Scalar OP Vector                                            
         RETURN output;
         const auto same_lhs = static_cast<LOSSLESS>(DenseCast(GetFirst(lhs)));
         for (Count i = 0; i < S; ++i) {
            output[i] = static_cast<LOSSLESS>(op(
               same_lhs,
               static_cast<LOSSLESS>(DenseCast(rhs[i]))
            ));
         }

         return output;
      }
      else {
         // Scalar OP Scalar                                            
         // Casts are no-op if types are the same                       
         return static_cast<LOSSLESS>(op(
            static_cast<LOSSLESS>(DenseCast(GetFirst(lhs))),
            static_cast<LOSSLESS>(DenseCast(GetFirst(rhs)))
         ));
      }
   }

   /// Attempt register encapsulation of LHS and RHS arrays                   
   /// Check if result of opSIMD is supported and return it, otherwise        
   /// fallback to opFALL and calculate conventionally                        
   ///   @tparam DEF - default value to fill empty register regions           
   ///                 useful against division-by-zero cases                  
   ///   @tparam REGISTER - the register to use for the SIMD operation        
   ///   @tparam OUT - the type of data we want as a result                   
   ///   @param lhs - left argument                                           
   ///   @param rhs - right argument                                          
   ///   @param opSIMD - the SIMD function to invoke                          
   ///   @param opFALL - the fallback function to invoke, if SIMD is not      
   ///      supported                                                         
   ///   @return the result (either std::array, number, or register)          
   template<auto DEF, class REGISTER, class OUT, class LHS, class RHS, class FSIMD, class FFALL>
   NOD() LANGULUS(INLINED)
   constexpr auto Evaluate2(const LHS& lhs, const RHS& rhs, FSIMD&& opSIMD, FFALL&& opFALL) {
      using OUTSIMD = InvocableResult2<FSIMD, REGISTER>;
      constexpr auto S = OverlapCounts<LHS, RHS>();
      LANGULUS_SIMD_VERBOSE_TAB("Evaluated to overlapped count of ", S);

      if constexpr (S < 2 or CT::NotSIMD<REGISTER> or CT::NotSIMD<OUTSIMD>) {
         // Call the fallback routine if unsupported, or size 1         
         return Fallback2<OUT>(lhs, rhs, ::std::move(opFALL));
      }
      else if constexpr (CT::Vector<LHS> and CT::Vector<RHS>) {
         // Both LHS and RHS are vectors, so wrap in registers          
         LANGULUS_SIMD_VERBOSE("Both sides are vectors");
         return opSIMD(
            Convert<DEF, OUT>(lhs),
            Convert<DEF, OUT>(rhs)
         );
      }
      else if constexpr (CT::Vector<LHS>) {
         // LHS is vector, RHS is either already register, or scalar    
         if constexpr (CT::SIMD<RHS>) {
            return opSIMD(
               Convert<DEF, OUT>(lhs),
               rhs
            );
         }
         else {
            return opSIMD(
               Convert<DEF, OUT>(lhs),
               Fill<REGISTER, OUT>(rhs)
            );
         }
      }
      else if constexpr (CT::Vector<RHS>) {
         // LHS is either scalar or register, RHS is vector             
         if constexpr (CT::SIMD<LHS>) {
            return opSIMD(
               lhs,
               Convert<DEF, OUT>(rhs)
            );
         }
         else {
            return opSIMD(
               Fill<REGISTER, OUT>(lhs),
               Convert<DEF, OUT>(rhs)
            );
         }
      }
      else if constexpr (CT::SIMD<LHS, RHS>) {
         // Both LHS and RHS are already registers                      
         return opSIMD(lhs, rhs);
      }
      else {
         // Both LHS and RHS are scalars (or anything else)             
         // Just fallback and use the appropriate operator              
         return Fallback2<OUT>(lhs, rhs, ::std::move(opFALL));
      }
   }

} // namespace Langulus::SIMD::Inner
