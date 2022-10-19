#include "Main.hpp"
#include <catch2/catch.hpp>

template<class LHS, class RHS, class OUT>
LANGULUS(ALWAYSINLINE) void ControlDiv(const LHS& lhs, const RHS& rhs, OUT& out) {
	if (DenseCast(rhs) == Decay<RHS> {0})
		LANGULUS_THROW(DivisionByZero, "Division by zero");

	DenseCast(out) = DenseCast(lhs) / DenseCast(rhs);
}

template<class LHS, class RHS, size_t C, class OUT>
LANGULUS(ALWAYSINLINE) void ControlDiv(const Vector<LHS, C>& lhsArray, const Vector<RHS, C>& rhsArray, Vector<OUT, C>& out) {
	auto r = out.mArray;
	auto lhs = lhsArray.mArray;
	auto rhs = rhsArray.mArray;
	const auto lhsEnd = lhs + C;
	while (lhs != lhsEnd)
		ControlDiv(*lhs++, *rhs++, *r++);
}

TEMPLATE_TEST_CASE("Divide", "[divide]"
	, NUMBERS_ALL()
	, VECTORS_ALL(1)
	, VECTORS_ALL(2)
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

	GIVEN("x / y = r") {
		T x, y;
		T r, rCheck;

		if constexpr (!CT::Typed<T>) {
			if constexpr (CT::Sparse<T>) {
				x = nullptr;
				y = nullptr;
				r = new Decay<T>;
				rCheck = new Decay<T>;
			}

			InitOne(x, 1);
			InitOne(y, -5);
		}

		WHEN("Divided") {
			ControlDiv(x, y, rCheck);
			if constexpr (CT::Typed<T>)
				SIMD::Divide(x.mArray, y.mArray, r.mArray);
			else
				SIMD::Divide(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(DenseCast(r) == DenseCast(rCheck));
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Divide (control)") (timer meter) {
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
						ControlDiv(nx[i], ny[i], nr[i]);
					});
				};

				BENCHMARK_ADVANCED("Divide (SIMD)") (timer meter) {
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
							SIMD::Divide(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						else
							SIMD::Divide(nx[i], ny[i], nr[i]);
					});
				};
			#endif
		}

		WHEN("Divided in reverse") {
			ControlDiv(y, x, rCheck);
			if constexpr (CT::Typed<T>)
				SIMD::Divide(y.mArray, x.mArray, r.mArray);
			else
				SIMD::Divide(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(DenseCast(r) == DenseCast(rCheck));
			}
		}

		WHEN("Divided by zero") {
			if constexpr (!CT::Typed<T>)
				InitOne(x, 0);
			else
				DenseCast(x.mArray[0]) = {};

			REQUIRE_THROWS(ControlDiv(y, x, rCheck));
			if constexpr (CT::Typed<T>)
				REQUIRE_THROWS(SIMD::Divide(y.mArray, x.mArray, r.mArray));
			else
				REQUIRE_THROWS(SIMD::Divide(y, x, r));
		}

		if constexpr (CT::Sparse<T>) {
			delete r;
			delete rCheck;
			delete x;
			delete y;
		}
	}
}