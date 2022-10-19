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
	Byte PTR, \
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
	(Vector<Byte PTR, S>), \
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
		if (a)
			delete a;
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
				if (*i == TD {0})
					*i = TD {1};
			}
			else {
				i = static_cast<T>(gen() % 66);
				if (i == T {0})
					i = T {1};
			}
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
