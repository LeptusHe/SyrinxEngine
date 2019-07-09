#pragma once
#include <unordered_map>
#include "Common/SyrinxAssert.h"

namespace Syrinx {

template <typename Key, typename Value>
class Cache {
public:
    Cache() = default;
    ~Cache() = default;

    void add(const Key& key, Value *value);
    void add(const Key& key, std::unique_ptr<Value>&& value);
    bool remove(const Key& key);
    Value* find(const Key& key) const;

private:
    std::unordered_map<Key, std::unique_ptr<Value>> mMap;
};


template <typename Key, typename Value>
void Cache<Key, Value>::add(const Key& key, Value* value)
{
    add(key, std::unique_ptr<Value>(value));
}


template <typename Key, typename Value>
void Cache<Key, Value>::add(const Key& key, std::unique_ptr<Value>&& value)
{
    SYRINX_EXPECT(value);
    SYRINX_EXPECT(!find(key));
    mMap.insert({key, std::move(value)});
    SYRINX_ENSURE(find(key));
}


template <typename Key, typename Value>
bool Cache<Key, Value>::remove(const Key& key)
{
    auto iter = mMap.find(key);
    if (iter == std::end(mMap)) {
        return false;
    }
    mMap.erase(key);
    SYRINX_ENSURE(!find(key));
    return true;
}


template <typename Key, typename Value>
Value* Cache<Key, Value>::find(const Key& key) const
{
    auto iter = mMap.find(key);
    if (iter == std::end(mMap)) {
        return nullptr;
    }
    return iter->second.get();
}

} // namespace Syrinx