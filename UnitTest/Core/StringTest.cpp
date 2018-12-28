#include <gmock/gmock.h>
#include <Container/String.h>

using namespace testing;
using namespace Syrinx;


TEST(String, to_upper)
{
    std::string testCase = "string123";

    ASSERT_THAT(ToUpper(testCase), Eq("STRING123"));
}


TEST(String, split_string_by_space)
{
    const std::vector<std::string> stringList = {"split", "string", "by", "space"};
    std::string text;
    for (const auto& str : stringList) {
        text += str + " ";
    }

    auto resultList = SplitStringBySpace(text);
    ASSERT_THAT(resultList.size(), Eq(stringList.size()));
    for (int i = 0; i < resultList.size(); ++ i) {
        ASSERT_THAT(resultList[i], Eq(stringList[i]));
    }
}


TEST(String, parse_float_array)
{
    const std::string floatStringList = "1.234 1  2.2   4.0";
    const std::vector<float> expectedFloatList = {1.234, 1.0, 2.2, 4.0};
    auto resultList = ParseFloatArray(floatStringList);

    ASSERT_THAT(resultList.size(), Eq(expectedFloatList.size()));
    for (int i = 0; i < resultList.size(); ++ i) {
        ASSERT_FLOAT_EQ(resultList[i], expectedFloatList[i]);
    }
}