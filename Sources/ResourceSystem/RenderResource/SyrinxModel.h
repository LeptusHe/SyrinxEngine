#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include "RenderResource/SyrinxMesh.h"
#include "RenderResource/SyrinxMaterial.h"
#include "RenderResource/SyrinxRenderResource.h"
#include "ResourceManager/SyrinxFileManager.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

class Model : public RenderResource {
public:
    using MeshMap = std::unordered_map<std::string, Mesh*>;
    using MeshList = std::vector<Mesh*>;
    using MaterialMap = std::unordered_map<std::string, Material*>;
    using MaterialList = std::vector<Material*>;
    using MeshMaterialPair = std::pair<Mesh*, Material*>;
    using MeshMaterialPairMap = std::unordered_map<std::string, MeshMaterialPair>;
    using MeshMaterialPairList = std::vector<MeshMaterialPair>;

public:
    explicit Model(const std::string& name);
    ~Model() override = default;

    void addMeshMaterialPair(const MeshMaterialPair& meshMaterialPair);
    const Mesh* getMesh(const std::string& name) const;
    const MeshMap& getMeshMap() const;
    const MeshList& getMeshList() const;
    const Material* getMaterial(const std::string& name) const;
    const MaterialMap& getMaterialMap() const;
    const MaterialList& getMaterialList() const;
    const MeshMaterialPairMap& getMeshMaterialPairMap() const;
    const MeshMaterialPairList& getMeshMaterialPairList() const;

private:
    void addMesh(Mesh* mesh);
    void addMaterial(Material *material);

private:
    MeshMap mMeshMap;
    MeshList mMeshList;
    MaterialMap mMaterialMap;
    MaterialList mMaterialList;
    MeshMaterialPairMap mMeshMaterialPairMap;
    MeshMaterialPairList mMeshMaterialPairList;
};

} // namespace Syrinx