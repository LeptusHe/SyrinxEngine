#pragma once
#include <cuda_runtime.h>

namespace Syrinx {

class CudaTexture {
public:
    void init();

private:
    cudaTextureObject_t mHandle;
};

} // namespace Syrinx