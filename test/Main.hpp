#pragma once
#include <LangulusSIMD.hpp>
#include <cstdint>
#include <cstddef>
#include <random>

using namespace Langulus;

//#define LANGULUS_STD_BENCHMARK

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#define NUMBERS_SIGNED(PTR) \
	::std::int8_t PTR, \
	::std::int16_t PTR, \
	::std::int32_t PTR, \
	::std::int64_t PTR, \
	float PTR, double PTR

#define NUMBERS_UNSIGNED(PTR) \
	::std::uint8_t PTR, \
	::std::uint16_t PTR, \
	::std::uint32_t PTR, \
	::std::uint64_t PTR, \
	::std::byte PTR, \
	char8_t PTR, char16_t PTR, char32_t PTR, wchar_t PTR

#define NUMBERS_ALL() \
	NUMBERS_SIGNED( ), \
	NUMBERS_UNSIGNED( ), \
	NUMBERS_SIGNED(*), \
	NUMBERS_UNSIGNED(*)

#define VECTORS_SIGNED(PTR,S) \
	(Vector<::std::int8_t PTR, S>), \
	(Vector<::std::int16_t PTR, S>), \
	(Vector<::std::int32_t PTR, S>), \
	(Vector<::std::int64_t PTR, S>), \
	(Vector<float PTR, S>), \
	(Vector<double PTR, S>)

#define VECTORS_UNSIGNED(PTR,S) \
	(Vector<::std::uint8_t PTR, S>), \
	(Vector<::std::uint16_t PTR, S>), \
	(Vector<::std::uint32_t PTR, S>), \
	(Vector<::std::uint64_t PTR, S>), \
	(Vector<::std::byte PTR, S>), \
	(Vector<char8_t PTR, S>), \
	(Vector<char16_t PTR, S>), \
	(Vector<char32_t PTR, S>), \
	(Vector<wchar_t PTR, S>)

#define VECTORS_ALL(S) \
	VECTORS_SIGNED(*,S), \
	VECTORS_UNSIGNED(*,S), \
	VECTORS_SIGNED( ,S), \
	VECTORS_UNSIGNED( ,S)

using uint = unsigned int;
template<class T>
using some = std::vector<T>;

template<class T, class A>
void InitOne(T& a, A&& b) noexcept {
	if constexpr (CT::Sparse<T>) {
		using DT = Decay<T>;
		a = new DT {static_cast<DT>(b)};
	}
	else a = static_cast<T>(b);
}

template<class T, Count C>
struct alignas(Langulus::Alignment) Vector {
	// This makes the type CT::Typed
	using MemberType = T;

	T mArray[C];

	Vector() {
		static std::random_device rd;
		static std::mt19937 gen(rd());

		for (auto& i : mArray) {
			if constexpr (CT::Sparse<T>) {
				using TD = Decay<T>;
				i = new TD {static_cast<TD>(gen() % 66)};
			}
			else i = static_cast<T>(gen() % 66);
		}
	}

	~Vector() {
		for (auto& i : mArray) {
			if constexpr (CT::Sparse<T>)
				delete i;
		}
	}

	bool operator == (const Vector& e) const noexcept {
		for (int i = 0; i < C; ++i)
			if (DenseCast(mArray[i]) != DenseCast(e.mArray[i]))
				return false;
		return true;
	}
};

/*template<class LHS, class RHS, class OUT>
LANGULUS(ALWAYSINLINE) void Control(const LHS&, const RHS&, OUT&) noexcept;

template<class LHS, class RHS, size_t C, class OUT>
LANGULUS(ALWAYSINLINE) void Control(const Vector<LHS, C>&, const Vector<RHS, C>&, Vector<OUT, C>&) noexcept;

#define CONTROL_SNIPPETS(OP) \
	using timer = Catch::Benchmark::Chronometer; \
	template<class T> \
	using uninitialized = Catch::Benchmark::storage_for<T>; \
	template<class LHS, class RHS, class OUT> \
	LANGULUS(ALWAYSINLINE) void Control(const LHS& lhs, const RHS& rhs, OUT& out) noexcept { \
		if constexpr (CT::Same<OUT, ::std::byte>) { \
			DenseCast(out) = static_cast<Decay<OUT>>( \
				reinterpret_cast<const unsigned char&>(DenseCast(lhs)) OP \
				reinterpret_cast<const unsigned char&>(DenseCast(rhs)) \
			); \
		} \
		else DenseCast(out) = DenseCast(lhs) OP DenseCast(rhs); \
	} \
	template<class LHS, class RHS, size_t C, class OUT> \
	LANGULUS(ALWAYSINLINE) void Control(const Vector<LHS, C>& lhsArray, const Vector<RHS, C>& rhsArray, Vector<OUT, C>& out) noexcept { \
		auto r = out.mArray; \
		auto lhs = lhsArray.mArray; \
		auto rhs = rhsArray.mArray; \
		const auto lhsEnd = lhs + C; \
		while (lhs != lhsEnd) { \
			if constexpr (CT::Same<OUT, ::std::byte>) { \
				DenseCast(*r) = static_cast<Decay<OUT>>( \
					reinterpret_cast<const unsigned char&>(DenseCast(*lhs)) OP \
					reinterpret_cast<const unsigned char&>(DenseCast(*rhs)) \
				); \
			} \
			else DenseCast(*r) = DenseCast(*lhs) OP DenseCast(*rhs); \
			++lhs; ++rhs; ++r; \
		} \
	}
	*/