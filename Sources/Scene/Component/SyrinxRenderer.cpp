#include "Component/SyrinxRenderer.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

Renderer::Renderer() : mMesh(nullptr), mMaterial(nullptr)
{
    SYRINX_ENSURE(!mMesh);
    SYRINX_ENSURE(!mMaterial);
}


void Renderer::setMesh(Mesh *mesh)
{
    mMesh = mesh;
    SYRINX_ENSURE(mMesh);
    SYRINX_ENSURE(mMesh == mesh);
}


void Renderer::setMaterial(Material *material)
{
    mMaterial = material;
    SYRINX_ENSURE(mMaterial);
    SYRINX_ENSURE(mMaterial == material);
}


const Mesh* Renderer::getMesh() const
{
    return mMesh;
}


Material* Renderer::getMaterial()
{
    return mMaterial;
}


bool Renderer::isValid() const
{
    return mMesh && mMaterial;
}

} // namespace Syrinx