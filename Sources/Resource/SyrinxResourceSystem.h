#pragma once
#include <memory>
#include "ResourceManager/SyrinxResourceManager.h"

namespace Syrinx {

class ResourceSystem {
public:
    using ResourceManagerMap = std::unordered_map<Manager::ManagerType, std::unique_ptr<Manager>>;

public:
    template <typename T> bool addResourceManager(std::unique_ptr<ResourceManager<T>>&& resourceManager);
    template <typename T> T* getResourceManager();
    template <typename T> const T* getResourceManager() const;

private:
    ResourceManagerMap mResourceManagerMap;
};



template <typename T>
bool ResourceSystem::addResourceManager(std::unique_ptr<ResourceManager<T>>&& resourceManager)
{
    SYRINX_EXPECT(resourceManager);
    auto managerType = resourceManager->getManagerType();
    SYRINX_EXPECT(!mResourceManagerMap.find(managerType));
    mResourceManagerMap[managerType] = std::move(resourceManager);
    SYRINX_ENSURE(mResourceManagerMap.find(managerType));
    return true;
}


} // namespace Syrinx