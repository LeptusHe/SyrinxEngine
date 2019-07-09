#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <FileSystem/SyrinxFileManager.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include "RenderResource/SyrinxMesh.h"
#include "RenderResource/SyrinxMaterial.h"
#include "RenderResource/SyrinxResource.h"

namespace Syrinx {

class Model : public Resource {
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
    Material* getMaterial(const std::string& name);
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