///                                                                           
/// Langulus::SIMD                                                            
/// Copyright (c) 2019 Dimo Markov <team@langulus.com>                        
/// Part of the Langulus framework, see https://langulus.com                  
///                                                                           
/// Distributed under GNU General Public License v3+                          
/// See LICENSE file, or https://www.gnu.org/licenses                         
///                                                                           
#include "Main.hpp"
#include <catch2/catch.hpp>

#if LANGULUS_SIMD(256BIT)

TEST_CASE("Strange clang-cl bug", "[bug]") {
   GIVEN("x + y = r") {
      __m256i x = _mm256_setr_epi64x(1, 2, 3, 4);
      __m256i y = _mm256_setr_epi64x(7, 8, 9, 0);
      __m256i control = _mm256_setr_epi64x(8, 10, 12, 4);

      WHEN("Added") {
         __m256i r = _mm256_add_epi64(x, y);

         THEN("The result should be correct") {
            alignas(32) std::uint64_t xx[4];
            alignas(32) std::uint64_t yy[4];
            alignas(32) std::uint64_t rr[4];
            alignas(32) std::uint64_t ctrl[4];

            _mm256_store_si256(reinterpret_cast<__m256i*>(xx), x);
            _mm256_store_si256(reinterpret_cast<__m256i*>(yy), y);
            _mm256_store_si256(reinterpret_cast<__m256i*>(rr), r);
            _mm256_store_si256(reinterpret_cast<__m256i*>(ctrl), control);

            REQUIRE(0 == memcmp(rr, ctrl, 32));
         }
      }
   }
}

#endif