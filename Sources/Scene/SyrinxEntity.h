#pragma once
#include <entityx/entityx.h>
#include <Common/SyrinxAssert.h>
#include "Component/SyrinxController.h"
#include "Component/SyrinxCamera.h"

namespace Syrinx {

class Entity {
public:
    using EntityHandle = entityx::Entity;
    template <typename T> using ComponentHandle = entityx::ComponentHandle<T>;

public:
    Entity(const std::string& name, EntityHandle handle);
    ~Entity() = default;

    template <typename T, typename ... Args> void addComponent(Args&& ... args);
    const std::string& getName() const;
    template <typename T> const T& getComponent() const;
    template <typename T> T& getComponent();
    template <typename T> bool hasComponent() const;
    void addController(Controller *controller);
    void addCamera(const Camera& camera);
    Controller* getController() const;
    Camera& getCamera() const;

private:
    std::string mName;
    EntityHandle mHandle;
};


template <typename T, typename ... Args>
void Entity::addComponent(Args&& ... args)
{
    mHandle.assign<T>(std::forward<Args>(args) ...);
}


template <typename T>
const T& Entity::getComponent() const
{
    const auto componentHandle = mHandle.component<T>();
    SYRINX_EXPECT(componentHandle);
    return *componentHandle;
}


template <typename T>
T& Entity::getComponent()
{
    auto componentHandle = mHandle.component<T>();
    SYRINX_ENSURE(componentHandle);
    return *componentHandle;
}


template <typename T>
bool Entity::hasComponent() const
{
    return mHandle.has_component<T>();
}

} // namespace Syrinx