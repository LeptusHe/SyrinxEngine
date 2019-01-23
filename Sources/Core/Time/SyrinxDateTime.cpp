#include "Time/SyrinxDateTime.h"
#include <chrono>
#include <ctime>
#include <iostream>

namespace Syrinx {

DateTime DateTime::now()
{
    auto timer = std::chrono::high_resolution_clock::now();
    auto secondsSinEpoch = std::chrono::duration_cast<std::chrono::seconds>(timer.time_since_epoch());
    auto millisecondsSinEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(timer.time_since_epoch());
    auto millisecond = static_cast<int>((millisecondsSinEpoch - secondsSinEpoch).count());

    std::time_t timeSinceEpoch = secondsSinEpoch.count();

#if defined(SYRINX_OS_WINDOWS)
    #define LOCALTIME(time, tm) localtime_s((&tm), (&time))
#elif defined(SYRINX_OS_LINUX)
    #define LOCALTIME(time, tm) localtime_r((&time), (&tm))
#else
    #define LOCALTIME(time, tm) localtime_r((&time), (&tm))
#endif

    struct tm time = {};
    LOCALTIME(timeSinceEpoch, time);

    return {time.tm_year + 1900, time.tm_mon + 1, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec, millisecond};
#undef LOCALTIME
}


DateTime::DateTime(int year, int month, int day, int hour, int minute, int second, int millisecond)
    : mYear(year)
    , mMonth(month)
    , mDay(day)
    , mHour(hour)
    , mMinute(minute)
    , mSecond(second)
    , mMillisecond(millisecond)
{

}

} // namespace Syrinx
