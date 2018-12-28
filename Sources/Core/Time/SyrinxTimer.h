#pragma once
#include <chrono>

namespace Syrinx {

class Timer {
public:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using SecondDuration = std::chrono::seconds;
    using MicrosecondDuration = std::chrono::microseconds;
    using MillisecondDuration = std::chrono::milliseconds;

public:
    Timer() = default;
    ~Timer() = default;

    Timer::TimePoint now();
    void start();
    float end(bool reset = true);
    template <typename T = MicrosecondDuration> T end(bool reset = true);

private:
    template <typename T = MicrosecondDuration> T getElapsedTime() const;
    float getElapsedSecond() const;

private:
    TimePoint mStartTime;
    TimePoint mEndTime;
};


template <typename T>
T Timer::end(bool reset)
{
    mEndTime = now();
    T elapsedSecond = getElapsedTime<T>();

    if (reset) {
        mStartTime = TimePoint();
        mEndTime = TimePoint();
    }
    return elapsedSecond;
}


template <typename T>
T Timer::getElapsedTime() const
{
    return std::chrono::duration_cast<T>(mEndTime - mStartTime);
}

} // namespace Syrinx