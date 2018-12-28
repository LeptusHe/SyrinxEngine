#pragma once
#include <better-enums/enum.h>
#include <Math/SyrinxMath.h>

namespace Syrinx {

struct DepthState {
    bool depthTestEnable = true;
    bool clearEnable = true;
    Color clearColor{0.5, 0.0, 0.5, 1.0};
    bool depthWriteEnable = true;
};


class RenderState {
public:
    DepthState depthState;
};

} // namespace Syrinx