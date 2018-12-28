#pragma once
#include <RenderResource/SyrinxMesh.h>
#include <RenderResource/SyrinxMaterial.h>

namespace Syrinx {

class Renderer {
public:
    Renderer();
    ~Renderer() = default;

    void setMesh(Mesh *mesh);
    void setMaterial(Material *material);
    const Mesh* getMesh() const;
    const Material* getMaterial() const;
    bool isValid() const;

private:
    Mesh *mMesh;
    Material *mMaterial;
};

} // namespace Syrinx