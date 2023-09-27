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
      const std::int64_t xsrc[3] {1,2,3};
      const std::int64_t ysrc[3] {7,8,9};
      const std::int64_t ctrlsrc[3] {8,10,12};

      const __m256i x = _mm256_setr_epi64x(xsrc[0], xsrc[1], xsrc[2], 0);
      const __m256i y = _mm256_setr_epi64x(ysrc[0], ysrc[1], ysrc[2], 0);
      const __m256i control = _mm256_setr_epi64x(ctrlsrc[0], ctrlsrc[1], ctrlsrc[2], 0);

      WHEN("Added") {
         const __m256i r = _mm256_add_epi64(x, y);

         alignas(32) std::uint64_t xx[4];
         alignas(32) std::uint64_t yy[4];
         alignas(32) std::uint64_t rr[4];
         alignas(32) std::uint64_t ctrl[4];

         _mm256_store_si256(reinterpret_cast<__m256i*>(xx), x);
         _mm256_store_si256(reinterpret_cast<__m256i*>(yy), y);
         _mm256_store_si256(reinterpret_cast<__m256i*>(rr), r);
         _mm256_store_si256(reinterpret_cast<__m256i*>(ctrl), control);

         THEN("The result should be correct") {
            REQUIRE(0 == memcmp(xx,    xsrc,    24));
            REQUIRE(0 == memcmp(yy,    ysrc,    24));
            REQUIRE(0 == memcmp(ctrl,  ctrlsrc, 24));
            REQUIRE(0 == memcmp(rr,    ctrl,    24));
         }
      }
   }
}

TEST_CASE("Strange clang-cl bug (simde equivalent)", "[bug]") {
   GIVEN("x + y = r") {
      const std::int64_t xsrc[3] {1,2,3};
      const std::int64_t ysrc[3] {7,8,9};
      const std::int64_t ctrlsrc[3] {8,10,12};

      const simde__m256i x = simde_mm256_setr_epi64x(xsrc[0], xsrc[1], xsrc[2], 0);
      const simde__m256i y = simde_mm256_setr_epi64x(ysrc[0], ysrc[1], ysrc[2], 0);
      const simde__m256i control = simde_mm256_setr_epi64x(ctrlsrc[0], ctrlsrc[1], ctrlsrc[2], 0);

      WHEN("Added") {
         const simde__m256i r = simde_mm256_add_epi64(x, y);

         alignas(32) std::uint64_t xx[4];
         alignas(32) std::uint64_t yy[4];
         alignas(32) std::uint64_t rr[4];
         alignas(32) std::uint64_t ctrl[4];

         simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(xx), x);
         simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(yy), y);
         simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(rr), r);
         simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(ctrl), control);

         THEN("The result should be correct") {
            REQUIRE(0 == memcmp(xx,    xsrc,    24));
            REQUIRE(0 == memcmp(yy,    ysrc,    24));
            REQUIRE(0 == memcmp(ctrl,  ctrlsrc, 24));
            REQUIRE(0 == memcmp(rr,    ctrl,    24));
         }
      }
   }
}

TEST_CASE("Strange clang-cl bug (langulus equivalent)", "[bug]") {
   GIVEN("x + y = r") {
      const std::int64_t xsrc[3] {1,2,3};
      const std::int64_t ysrc[3] {7,8,9};
      const std::int64_t ctrlsrc[3] {8,10,12};

      const simde__m256i x = SIMD::Set(xsrc);
      const simde__m256i y = SIMD::Set(ysrc);
      const simde__m256i control = SIMD::Set(ctrlsrc);

      WHEN("Added") {
         const simde__m256i r  = SIMD::Inner::Add<std::int64_t>(x, y);

         std::uint64_t xx[3];
         std::uint64_t yy[3];
         std::uint64_t rr[3];
         std::uint64_t ctrl[3];

         SIMD::Store(x, xx);
         SIMD::Store(y, yy);
         SIMD::Store(r, rr);
         SIMD::Store(control, ctrl);

         std::uint64_t rr2[3];
         SIMD::Add(xsrc, ysrc, rr2);

         THEN("The result should be correct") {
            REQUIRE(0 == memcmp(xx,    xsrc,    24));
            REQUIRE(0 == memcmp(yy,    ysrc,    24));
            REQUIRE(0 == memcmp(ctrl,  ctrlsrc, 24));
            REQUIRE(0 == memcmp(rr,    ctrl,    24));
            REQUIRE(0 == memcmp(rr2,   rr,      24));
         }
      }
   }
}

TEST_CASE("Strange clang-cl bug (langulus vector equivalent)", "[bug]") {
   GIVEN("x + y = r") {
      const Vector<std::int64_t, 3> xsrc;
      const Vector<std::int64_t, 3> ysrc;
      Vector<std::int64_t, 3> ctrlsrc1;
      Vector<std::int64_t, 3> ctrlsrc2;

      WHEN("Added") {
         SIMD::Add(xsrc.mArray, ysrc.mArray, ctrlsrc1.mArray);
         SIMD::Add(xsrc, ysrc, ctrlsrc2);

         THEN("The result should be correct") {
            REQUIRE(ctrlsrc1[0] == xsrc[0] + ysrc[0]);
            REQUIRE(ctrlsrc1[1] == xsrc[1] + ysrc[1]);
            REQUIRE(ctrlsrc1[2] == xsrc[2] + ysrc[2]);
            REQUIRE(ctrlsrc2[0] == xsrc[0] + ysrc[0]);
            REQUIRE(ctrlsrc2[1] == xsrc[1] + ysrc[1]);
            REQUIRE(ctrlsrc2[2] == xsrc[2] + ysrc[2]);
         }
      }
   }
}

#endif