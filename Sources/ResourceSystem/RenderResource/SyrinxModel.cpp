#include "RenderResource/SyrinxModel.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

Model::Model(const std::string& name)
    : RenderResource(name)
    , mMeshMap()
    , mMeshList()
    , mMaterialMap()
    , mMaterialList()
    , mMeshMaterialPairMap()
    , mMeshMaterialPairList()
{
    SYRINX_ENSURE(!getName().empty());
    SYRINX_ENSURE(mMeshMap.empty());
    SYRINX_ENSURE(mMeshList.empty());
    SYRINX_ENSURE(mMaterialMap.empty());
    SYRINX_ENSURE(mMaterialList.empty());
    SYRINX_ENSURE(mMeshMaterialPairMap.empty());
    SYRINX_ENSURE(mMeshMaterialPairList.empty());
}


void Model::addMeshMaterialPair(const Model::MeshMaterialPair& meshMaterialPair)
{
    SYRINX_EXPECT(meshMaterialPair.first && meshMaterialPair.second);
    Mesh *mesh = meshMaterialPair.first;
    Material *material = meshMaterialPair.second;
    addMesh(mesh);
    addMaterial(material);
    mMeshMaterialPairMap[mesh->getName()] = meshMaterialPair;
    mMeshMaterialPairList.push_back(meshMaterialPair);
}


void Model::addMesh(Mesh *mesh)
{
    SYRINX_EXPECT(mesh);
    SYRINX_EXPECT(!getMesh(mesh->getName()));
    mMeshMap[mesh->getName()] = mesh;
    mMeshList.push_back(mesh);
    SYRINX_ENSURE(getMesh(mesh->getName()) == mesh);
}


void Model::addMaterial(Material *material)
{
    SYRINX_EXPECT(material);
    SYRINX_EXPECT(!getMaterial(material->getName()));
    mMaterialMap[material->getName()] = material;
    mMaterialList.push_back(material);
    SYRINX_ENSURE(getMaterial(material->getName()) == material);
}


const Mesh* Model::getMesh(const std::string& name) const
{
    auto iter = mMeshMap.find(name);
    return (iter == std::end(mMeshMap)) ? nullptr : iter->second;
}


const Model::MeshMap& Model::getMeshMap() const
{
    return mMeshMap;
}


const Model::MeshList& Model::getMeshList() const
{
    return mMeshList;
}


const Material* Model::getMaterial(const std::string& name) const
{
    auto iter = mMaterialMap.find(name);
    if (iter == std::end(mMaterialMap)) {
        return nullptr;
    }
    return iter->second;
}


const Model::MaterialMap& Model::getMaterialMap() const
{
    return mMaterialMap;
}


const Model::MaterialList& Model::getMaterialList() const
{
    return mMaterialList;
}


const Model::MeshMaterialPairMap& Model::getMeshMaterialPairMap() const
{
    return mMeshMaterialPairMap;
}


const Model::MeshMaterialPairList& Model::getMeshMaterialPairList() const
{
    return mMeshMaterialPairList;
}

} // namespace Syrinx