#include "Scene/SyrinxEntity.h"
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>
#include <Input/SyrinxInput.h>

namespace Syrinx {

Entity::Entity(const std::string& name, EntityHandle handle)
    : mName(name)
    , mHandle(handle)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mHandle.valid());
}


const std::string& Entity::getName() const
{
    return mName;
}


void Entity::addController(Controller *controller)
{
    SYRINX_EXPECT(controller);
    if (!hasComponent<Transform>()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState,
                "fail to add controller to entity [{}] because it does not have transform component", getName());
    }
    auto& transform = getComponent<Transform>();
    controller->setTransform(&transform);
    controller->setInput(Input::getInstancePtr());
    controller->setEntity(this);
    mHandle.assign<Syrinx::Controller*>(controller);
}

} // namespace Syrinx