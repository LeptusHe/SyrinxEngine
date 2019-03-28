#pragma once
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <Common/SyrinxAssert.h>

namespace Syrinx {

class Manager {
public:
    using ManagerType = uint32_t;

protected:
    static uint32_t mManagerCounter;
};




template <typename T>
class ResourceManager : public Manager {
public:
    using Resource = T;
    using ResourceMap = std::unordered_map<std::string, std::unique_ptr<Resource>>;

public:
    static ManagerType getManagerType();

public:
    ResourceManager() = default;
    virtual ~ResourceManager() = default;

    Resource* createOrRetrieve(const std::string& name);
    virtual std::unique_ptr<T> create(const std::string& name) = 0;
    Resource* find(const std::string& name);
    Resource* add(std::unique_ptr<Resource>&& resource);

private:
    ResourceMap mResourceMap;
};




template <typename T>
Manager::ManagerType ResourceManager<T>::getManagerType()
{
    static ManagerType managerType = ++Manager::mManagerCounter;
    return managerType;
}


template <typename T>
typename ResourceManager<T>::Resource* ResourceManager<T>::createOrRetrieve(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    Resource *resource = find(name);
    if (!resource) {
        auto resourceCreated = create(name);
        resource = add(std::move(resourceCreated));
    }
    SYRINX_ENSURE(find(name));
    return resource;
}


template <typename T>
typename ResourceManager<T>::Resource* ResourceManager<T>::find(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mResourceMap.find(name);
    if (iter == std::end(mResourceMap)) {
        return nullptr;
    }
    return iter->second.get();
}


template <typename T>
typename ResourceManager<T>::Resource* ResourceManager<T>::add(std::unique_ptr<Resource>&& resource)
{
    SYRINX_EXPECT(resource);
    SYRINX_ENSURE(!find(resource->getName()));
    Resource *result = resource.get();
    mResourceMap[resource->getName()] = std::move(resource);
    SYRINX_ENSURE(!resource);
    SYRINX_ENSURE(find(result->getName()));
    SYRINX_ENSURE(result);
    return result;
}

} // namespace Syrinx
