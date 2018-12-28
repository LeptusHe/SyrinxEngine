#include "Time/SyrinxTimer.h"

namespace Syrinx {

Timer::TimePoint Timer::now()
{
    return std::chrono::high_resolution_clock::now();
}


void Timer::start()
{
    mStartTime = now();
}


float Timer::end(bool reset)
{
    mEndTime = now();
    float elapsedSecond = getElapsedSecond();

    if (reset) {
        mStartTime = TimePoint();
        mEndTime = TimePoint();
    }
    return elapsedSecond;
}


float Timer::getElapsedSecond() const
{
    auto microsecondDuration = getElapsedTime<MicrosecondDuration>().count();
    return static_cast<float>(microsecondDuration) / static_cast<float>(1000);
}

} // namespace Syrinx
