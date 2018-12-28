#include <gmock/gmock.h>
#include <HardwareResource/SyrinxHardwareTextureSampler.h>

using namespace testing;
using namespace Syrinx;


TEST(HardwareTextureSampler, default_state)
{
    Syrinx::HardwareTextureSampler textureSampler("texture sampler");

    ASSERT_THAT(textureSampler.getMinFilterMethod()._value, Eq(TextureMinFilterMethod::LINEAR));
    ASSERT_THAT(textureSampler.getMagFilterMethod()._value, Eq(TextureMinFilterMethod::LINEAR));
}
