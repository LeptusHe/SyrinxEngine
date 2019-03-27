#pragma once
#include <string>
#include <vector>
#include <Math/SyrinxMath.h>

namespace Syrinx {

class UVChannel {
public:
    UVChannel(uint8_t numElement, const float *uvSet);
    ~UVChannel();
    UVChannel(const UVChannel&) = delete;
    UVChannel& operator=(const UVChannel&) = delete;

public:
    const uint8_t numElement;
    const float *uvSet;
};


class MeshGeometry {
public:
    MeshGeometry();
    MeshGeometry(const std::string& meshName, uint32_t vertexNumber,
                 Point3f *positions, Normal3f *normals, Normal3f *tangents, Normal3f *bitangents,
                 const std::vector<UVChannel*>& uvChannels, uint32_t triangleNumber, uint32_t *indices);
    MeshGeometry(const MeshGeometry& meshGeometry);
    MeshGeometry(MeshGeometry&& meshGeometry) noexcept;
    MeshGeometry& operator=(const MeshGeometry& meshGeometry);
    MeshGeometry& operator=(MeshGeometry&& meshGeometry) noexcept;
    ~MeshGeometry();

    bool operator==(const MeshGeometry& rhs) const;
    void swap(MeshGeometry& rhs) noexcept;
    bool isValid() const;
    bool isClean() const;

private:
    void clear();

public:
    std::string name;
    uint32_t numVertex;
    Point3f *positionSet;
    Normal3f *normalSet;
    Normal3f *tangentSet;
    Normal3f *bitangentSet;
    std::vector<UVChannel*> uvChannelSet;
    uint32_t numTriangle;
    uint32_t *indexSet;

private:
#define SYRINX_IS_FLOAT_TYPE(expression) \
    static_assert((std::is_floating_point<decltype(expression)>::value) && (sizeof(decltype(expression)) == sizeof(float)), "the type of expression" #expression " is not float")

    SYRINX_IS_FLOAT_TYPE(Point3f::x);
    SYRINX_IS_FLOAT_TYPE(Point3f::y);
    SYRINX_IS_FLOAT_TYPE(Point3f::z);
    SYRINX_IS_FLOAT_TYPE(Normal3f::x);
    SYRINX_IS_FLOAT_TYPE(Normal3f::y);
    SYRINX_IS_FLOAT_TYPE(Normal3f::z);

#undef SYRINX_IS_FLOAT_TYPE
};

extern void swap(MeshGeometry& lhs, MeshGeometry& rhs) noexcept;

} // namespace Syrinx


namespace std {

template <> void swap<Syrinx::MeshGeometry>(Syrinx::MeshGeometry& lhs, Syrinx::MeshGeometry& rhs) noexcept;

} // namespace std
