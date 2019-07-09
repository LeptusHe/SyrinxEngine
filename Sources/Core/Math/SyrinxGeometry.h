#pragma once
#include <type_traits>

namespace Syrinx {

template <typename T>
struct Offset2D {
    Offset2D() : x(static_cast<T>(0)), y(static_cast<T>(0)) {}
    Offset2D(const T& x_, const T& y_) : x(x_), y(y_) {}

    T x = static_cast<T>(0);
    T y = static_cast<T>(0);

    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value);
};


template <typename T>
struct Extent2D {
    Extent2D() : x(static_cast<T>(0)), y(static_cast<T>(0)) {}
    Extent2D(const T& x_, const T& y_) : x(x_), y(y_) {}

    T x = static_cast<T>(0);
    T y = static_cast<T>(0);

    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value);
};


template <typename T>
struct Rect2D {
    Offset2D<T> offset;
    Extent2D<T> extent;
};


} // namespace Syrinx