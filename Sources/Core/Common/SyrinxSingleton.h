#pragma once
#include "Common/SyrinxAssert.h"

namespace Syrinx {

template <typename T>
class Singleton {
public:
	static T& getInstance()
	{
		SYRINX_ASSERT(mSingleton);
		return (*mSingleton);
	}

	static T* getInstancePtr()
	{
		SYRINX_ASSERT(mSingleton);
		return mSingleton;
	}

public:
	Singleton()
	{
		SYRINX_ASSERT(!mSingleton);
		mSingleton = static_cast<T*>(this);
	}

	~Singleton()
	{
		SYRINX_ASSERT(mSingleton);
		mSingleton = nullptr;
	}

	Singleton(const Singleton<T>&) = delete;
	Singleton(const Singleton<T>&&) = delete;
	Singleton& operator=(const Singleton<T>&) = delete;
	Singleton& operator=(const Singleton<T>&&) = delete;

protected:
	static T *mSingleton;
};


template <typename T>
T *Singleton<T>::mSingleton = nullptr;;

} // namespace Syrinx