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


uint32_t Manager::mManagerCounter = 0;



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

private:
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
    auto resource = find(name);
    if (!resource) {
        resource = create(name);
        add(resource);
    }
    SYRINX_ENSURE(find(name));
    return resource.get();
}


template <typename T>
typename ResourceManager<T>::Resource* ResourceManager<T>::find(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mResourceMap.find(name);
    if (iter != std::end(mResourceMap)) {
        return iter->second.get();
    }
    return nullptr;
}


template <typename T>
typename ResourceManager<T>::Resource* ResourceManager<T>::add(std::unique_ptr<Resource>&& resource)
{
    SYRINX_EXPECT(resource);
    SYRINX_ENSURE(!find(resource->getName()));
    mResourceMap[resource->getName()] = std::move(resource);
    SYRINX_ENSURE(find(resource->getName()));
}

} // namespace Syrinx
