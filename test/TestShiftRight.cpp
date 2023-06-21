#include "Main.hpp"
#include <catch2/catch.hpp>

namespace Catch {
	template<>
	struct StringMaker<char8_t> {
		static std::string convert(char8_t const& value) {
			return std::to_string(static_cast<int>(value));
		}
	};

	template<>
	struct StringMaker<char16_t> {
		static std::string convert(char16_t const& value) {
			return std::to_string(static_cast<int>(value));
		}
	};

	template<>
	struct StringMaker<wchar_t> {
		static std::string convert(wchar_t const& value) {
			return std::to_string(static_cast<int>(value));
		}
	};

	template<>
	struct StringMaker<::Langulus::Byte> {
		static std::string convert(::Langulus::Byte const& value) {
			return std::to_string(static_cast<int>(value.mValue));
		}
	};
}

using timer = Catch::Benchmark::Chronometer;

template<class T>
using uninitialized = Catch::Benchmark::storage_for<T>;

template<class LHS, class RHS, class OUT>
LANGULUS(INLINED) void ControlSR(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
	static_assert(CT::IntegerX<Decay<LHS>, Decay<RHS>>, "Can only shift integers");
	// Well defined condition in SIMD calls, that is otherwise				
	// undefined behavior by C++ standard											
	DenseCast(out) = DenseCast(rhs) < Decay<RHS> {sizeof(Decay<RHS>) * 8} && DenseCast(rhs) >= 0
		? DenseCast(lhs) >> DenseCast(rhs) : 0;
}

template<class LHS, class RHS, size_t C, class OUT>
LANGULUS(INLINED) void ControlSR(const Vector<LHS, C>& lhsArray, const Vector<RHS, C>& rhsArray, Vector<OUT, C>& out) noexcept {
	static_assert(CT::IntegerX<Decay<LHS>, Decay<RHS>>, "Can only shift integers");
	auto r = out.mArray;
	auto lhs = lhsArray.mArray;
	auto rhs = rhsArray.mArray;
	const auto lhsEnd = lhs + C;
	while (lhs != lhsEnd)
		ControlSR(*lhs++, *rhs++, *r++);
}

TEMPLATE_TEST_CASE("Shift right", "[shift]"
	, NUMBERS_INT()
	, VECTORS_INT(1)
	, VECTORS_INT(2)
	, VECTORS_INT(3)
	, VECTORS_INT(4)
	, VECTORS_INT(5)
	, VECTORS_INT(8)
	, VECTORS_INT(9)
	, VECTORS_INT(16)
	, VECTORS_INT(17)
	, VECTORS_INT(32)
	, VECTORS_INT(33)
) {
	using T = TestType;

	GIVEN("x >> y = r") {
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

		WHEN("Shifted right") {
			ControlSR(x, y, rCheck);

			if constexpr (CT::Typed<T>)
				SIMD::ShiftRight(x.mArray, y.mArray, r.mArray);
			else
				SIMD::ShiftRight(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(DenseCast(r) == DenseCast(rCheck));
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Shifted right (control)") (timer meter) {
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
						ControlSR(nx[i], ny[i], nr[i]);
					});
				};

				BENCHMARK_ADVANCED("Shifted right (SIMD)") (timer meter) {
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
							SIMD::ShiftRight(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						else
							SIMD::ShiftRight(nx[i], ny[i], nr[i]);
					});
				};
			#endif
		}

		WHEN("Shifted right in reverse") {
			ControlSR(y, x, rCheck);

			if constexpr (CT::Typed<T>)
				SIMD::ShiftRight(y.mArray, x.mArray, r.mArray);
			else
				SIMD::ShiftRight(y, x, r);

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