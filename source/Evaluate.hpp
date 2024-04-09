#pragma once
#include "Convert.hpp"
#include "Bitmask.hpp"


namespace Langulus::SIMD::Inner
{

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
   constexpr auto Fallback(LHS& lhs, RHS& rhs, FFALL&& op) {
      using LOSSLESS = Decay<TypeOf<Lossless<LHS, RHS>>>;
      using RESULT = InvocableResult<FFALL, LOSSLESS>;
      constexpr auto S = OverlapCounts<LHS, RHS>();
      using RETURN = Conditional<
         CT::Bitmask<Desem<OUT>> or CT::Bool<RESULT>,
         Bitmask<S>, ::std::array<LOSSLESS, S>
      >;

      if constexpr (CT::Vector<LHS> and CT::Vector<RHS>) {
         // Vector OP Vector                                            
         RETURN output;
         for (Count i = 0; i < S; ++i) {
            output[i] = op(
               static_cast<LOSSLESS>(DenseCast(lhs[i])),
               static_cast<LOSSLESS>(DenseCast(rhs[i]))
            );
         }

         return output;
      }
      else if constexpr (CT::Vector<LHS>) {
         // Vector OP Scalar                                            
         RETURN output;
         const auto same_rhs = static_cast<LOSSLESS>(DenseCast(GetFirst(rhs)));
         for (Count i = 0; i < S; ++i) {
            output[i] = op(
               static_cast<LOSSLESS>(DenseCast(lhs[i])),
               same_rhs
            );
         }

         return output;
      }
      else if constexpr (CT::Vector<RHS>) {
         // Scalar OP Vector                                            
         RETURN output;
         const auto same_lhs = static_cast<LOSSLESS>(DenseCast(GetFirst(lhs)));
         for (Count i = 0; i < S; ++i) {
            output[i] = op(
               same_lhs,
               static_cast<LOSSLESS>(DenseCast(rhs[i]))
            );
         }

         return output;
      }
      else {
         // Scalar OP Scalar                                            
         // Casts are no-op if types are the same                       
         return op(
            static_cast<LOSSLESS>(DenseCast(GetFirst(lhs))),
            static_cast<LOSSLESS>(DenseCast(GetFirst(rhs)))
         );
      }
   }

   /// Attempt register encapsulation of LHS and RHS arrays                   
   /// Check if result of opSIMD is supported and return it, otherwise        
   /// fallback to opFALL and calculate conventionally                        
   ///   @tparam DEF - default value to fill empty register regions           
   ///                 useful against division-by-zero cases                  
   ///   @tparam REGISTER - the register to use for the SIMD operation        
   ///   @tparam OUT - the type of data we want as a result                   
   ///   @tparam LHS - left number type (deducible)                           
   ///   @tparam RHS - right number type (deducible)                          
   ///   @tparam FSIMD - the SIMD operation to invoke (deducible)             
   ///   @tparam FFALL - the fallback operation to invoke (deducible)         
   ///   @param lhs - left argument                                           
   ///   @param rhs - right argument                                          
   ///   @param opSIMD - the function to invoke                               
   ///   @param opFALL - the function to invoke                               
   ///   @return the result (either std::array, number, or register)          
   template<auto DEF, class REGISTER, class OUT, class LHS, class RHS, class FSIMD, class FFALL>
   NOD() LANGULUS(INLINED)
   constexpr auto Evaluate(const LHS& lhs, const RHS& rhs, FSIMD&& opSIMD, FFALL&& opFALL) {
      using OUTSIMD = InvocableResult<FSIMD, REGISTER>;
      constexpr auto S = OverlapCounts<LHS, RHS>();
      LANGULUS_SIMD_VERBOSE_TAB("Evaluated to overlapped count of ", S);

      if constexpr (S < 2 or CT::NotSIMD<REGISTER> or CT::NotSIMD<OUTSIMD>) {
         // Call the fallback routine if unsupported, or size 1         
         return Fallback<OUT>(lhs, rhs, ::std::move(opFALL));
      }
      else if constexpr (CT::Vector<LHS> and CT::Vector<RHS>) {
         // Both LHS and RHS are vectors, so wrap in registers          
         LANGULUS_SIMD_VERBOSE("Both sides are vectors");
         return opSIMD(
            Inner::Convert<DEF, OUT>(lhs),
            Inner::Convert<DEF, OUT>(rhs)
         );
      }
      else if constexpr (CT::Vector<LHS>) {
         // LHS is vector, RHS is scalar                                
         return opSIMD(
            Inner::Convert<DEF, OUT>(lhs),
            Fill<REGISTER, OUT>(rhs)
         );
      }
      else if constexpr (CT::Vector<RHS>) {
         // LHS is scalar, RHS is vector                                
         return opSIMD(
            Fill<REGISTER, OUT>(lhs),
            Inner::Convert<DEF, OUT>(rhs)
         );
      }
      else {
         // Both LHS and RHS are scalars (or anything else)             
         // Just fallback and use the appropriate operator              
         return Fallback<OUT>(lhs, rhs, ::std::move(opFALL));
      }
   }

} // namespace Langulus::SIMD::Inner
