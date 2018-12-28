#include "Container/String.h"
#include <sstream>
#include <iterator>

namespace Syrinx {

std::vector<std::string> SplitStringBySpace(const std::string& text)
{
    std::stringstream stringStream(text);
    return { std::istream_iterator<std::string>{stringStream}, std::istream_iterator<std::string>{} };
}


std::vector<float> ParseFloatArray(const std::string& text)
{
    auto floatStringList = SplitStringBySpace(text);
    std::vector<float> floatValueList;
    std::transform(std::begin(floatStringList), std::end(floatStringList), std::back_inserter(floatValueList), [](const std::string& floatString) {
        return std::stof(floatString);
    });
    return floatValueList;
}

} // namespace Syrinx