#include "Main.hpp"
#include <catch2/catch.hpp>

template<class LHS, class RHS, class OUT>
LANGULUS(ALWAYSINLINE) void ControlMul(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
	if constexpr (CT::Same<OUT, ::std::byte>) {
		DenseCast(out) = static_cast<Decay<OUT>>(
			reinterpret_cast<const unsigned char&>(DenseCast(lhs)) *
			reinterpret_cast<const unsigned char&>(DenseCast(rhs))
		);
	}
	else DenseCast(out) = DenseCast(lhs) * DenseCast(rhs);
}

template<class LHS, class RHS, size_t C, class OUT>
LANGULUS(ALWAYSINLINE) void ControlMul(const Vector<LHS, C>& lhsArray, const Vector<RHS, C>& rhsArray, Vector<OUT, C>& out) noexcept {
	auto r = out.mArray;
	auto lhs = lhsArray.mArray;
	auto rhs = rhsArray.mArray;
	const auto lhsEnd = lhs + C;
	while (lhs != lhsEnd) {
		if constexpr (CT::Same<OUT, ::std::byte>) {
			DenseCast(*r) = static_cast<Decay<OUT>>(
				reinterpret_cast<const unsigned char&>(DenseCast(*lhs)) *
				reinterpret_cast<const unsigned char&>(DenseCast(*rhs))
			);
		}
		else DenseCast(*r) = DenseCast(*lhs) * DenseCast(*rhs);
		++lhs; ++rhs; ++r;
	}
}

TEMPLATE_TEST_CASE("Multiply", "[multiply]"
	, VECTORS_ALL(2)
	, NUMBERS_ALL()
	, VECTORS_ALL(1)
	, VECTORS_ALL(3)
	, VECTORS_ALL(4)
	, VECTORS_ALL(5)
	, VECTORS_ALL(8)
	, VECTORS_ALL(9)
	, VECTORS_ALL(16)
	, VECTORS_ALL(17)
	, VECTORS_ALL(32)
	, VECTORS_ALL(33)
) {
	using T = TestType;

	GIVEN("x * y = r") {
		T x, y;
		T r, rCheck;

		if constexpr (!CT::Typed<T>) {
			if constexpr (CT::Sparse<T>) {
				r = new Decay<T>;
				rCheck = new Decay<T>;
			}

			InitOne(x, 1);
			InitOne(y, -5);
		}

		WHEN("Multiplied") {
			ControlMul(x, y, rCheck);
			if constexpr (CT::Typed<T>)
				SIMD::Multiply(x.mArray, y.mArray, r.mArray);
			else
				SIMD::Multiply(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(DenseCast(r) == DenseCast(rCheck));
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<T> nx(meter.runs());
					if constexpr (!CT::Typed<T>) {
						for (auto& i : nx)
							InitOne(i, 1);
					}

					some<T> ny(meter.runs());
					if constexpr (!CT::Typed<T>) {
						for (auto& i : ny)
							InitOne(i, 1);
					}

					some<T> nr(meter.runs());
					meter.measure([&](int i) {
						ControlMul(nx[i], ny[i], nr[i]);
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<T> nx(meter.runs());
					if constexpr (!CT::Typed<T>) {
						for (auto& i : nx)
							InitOne(i, 1);
					}

					some<T> ny(meter.runs());
					if constexpr (!CT::Typed<T>) {
						for (auto& i : ny)
							InitOne(i, 1);
					}

					some<T> nr(meter.runs());
					meter.measure([&](int i) {
						if constexpr (CT::Typed<T>)
							SIMD::Multiply(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						else
							SIMD::Multiply(nx[i], ny[i], nr[i]);
					});
				};
			#endif
		}

		WHEN("Multiplied in reverse") {
			ControlMul(y, x, rCheck);
			if constexpr (CT::Typed<T>)
				SIMD::Multiply(x.mArray, y.mArray, r.mArray);
			else
				SIMD::Multiply(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(DenseCast(r) == DenseCast(rCheck));
			}
		}

		if constexpr (CT::Sparse<T>) {
			delete r;
			delete rCheck;
			delete x;
			delete y;
		}
	}
}