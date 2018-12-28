#pragma once
#include <string>
#include "Common/SyrinxPlatform.h"

namespace Syrinx {

class DateTime {
public:
    static DateTime now();

public:
    ~DateTime() = default;
    int getYear() const  { return mYear; }
    int getMonth() const { return mMonth;}
    int getDay() const   { return mDay; }
    int getHour() const  { return mHour; }
    int getMinute() const { return mMinute; }
    int getSecond() const { return mSecond; }
    int getMillisecond() const { return mMillisecond; }

private:
    DateTime(int year, int month, int day, int hour, int minute, int second, int millisecond);

private:
    int mYear;
    int mMonth;
    int mDay;
    int mHour;
    int mMinute;
    int mSecond;
    int mMillisecond;
};

} // namespace Syrinx