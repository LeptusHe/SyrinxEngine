#pragma once
#include <string>
#include <vector>
#include <algorithm>

namespace Syrinx {

inline std::string ToUpper(const std::string& text)
{
    std::string result;
    std::transform(std::begin(text), std::end(text), std::back_inserter(result), ::toupper);
    return result;
}


inline std::string ToLower(const std::string& text)
{
    std::string result;
    std::transform(std::begin(text), std::end(text), std::back_inserter(result), ::tolower);
    return result;
}

std::vector<float> ParseFloatArray(const std::string& text);
std::vector<std::string> SplitStringBySpace(const std::string& text);

} // namespace Syrinx