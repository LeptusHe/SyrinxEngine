#pragma once
#include <vector>
#include <spirv_glsl.hpp>
#include "HardwareResource/SyrinxHardwareTexture.h"
#include "HardwareResource/SyrinxHardwareUniformBuffer.h"

namespace Syrinx {

using ReflectionType = spirv_cross::SPIRType;


template <typename T>
ReflectionType getReflectionTypeFromCppType()
{
    ReflectionType type;

#define cpp_type_to_ref_type(cpp_type, base_type) if (std::is_same<T, cpp_type>::value) { type.basetype = base_type; return type; }
    cpp_type_to_ref_type(bool, ReflectionType::BaseType::Boolean)
    cpp_type_to_ref_type(int, ReflectionType::BaseType::Int)
    cpp_type_to_ref_type(unsigned int, ReflectionType::BaseType::UInt)
    cpp_type_to_ref_type(int32_t, ReflectionType::BaseType::Int)
    cpp_type_to_ref_type(uint32_t, ReflectionType::BaseType::UInt)
    cpp_type_to_ref_type(float , ReflectionType::BaseType::Float)
    cpp_type_to_ref_type(double, ReflectionType::BaseType::Double)
#undef cpp_type_to_ref_type

#define cpp_vec_type_to_ref_type(cpp_type, base_type, vec_size) if (std::is_same<T, cpp_type>::value) { type.basetype = base_type; type.vecsize = vec_size; return type; }
    cpp_vec_type_to_ref_type(glm::vec2, ReflectionType::BaseType::Float, 2)
    cpp_vec_type_to_ref_type(glm::vec3, ReflectionType::BaseType::Float, 3)
    cpp_vec_type_to_ref_type(glm::vec4, ReflectionType::BaseType::Float, 4)
    cpp_vec_type_to_ref_type(glm::uvec2, ReflectionType::BaseType::UInt, 2)
    cpp_vec_type_to_ref_type(glm::uvec3, ReflectionType::BaseType::UInt, 3)
    cpp_vec_type_to_ref_type(glm::uvec4, ReflectionType::BaseType::UInt, 4)
    cpp_vec_type_to_ref_type(glm::ivec2, ReflectionType::BaseType::Int, 2)
    cpp_vec_type_to_ref_type(glm::ivec3, ReflectionType::BaseType::Int, 3)
    cpp_vec_type_to_ref_type(glm::ivec4, ReflectionType::BaseType::Int, 4)
#undef cpp_vec_type_to_ref_type

    if (std::is_same<T, glm::mat4>::value) {
        type.basetype = ReflectionType::BaseType::Float;
        type.vecsize = 4;
        type.columns = 4;
        return type;
    }

    SHOULD_NOT_GET_HERE();
    return type;
}


struct VariableInfo {
    std::string name;
    ReflectionType type;
    uint32_t size = 0;
};


struct TextureInfo : public VariableInfo {
    TextureType type = TextureType::UNDEFINED;
    uint32_t binding = 0;
};


struct InterfaceInfo : VariableInfo {
    uint32_t location = 0;
};


class UniformBufferInfo;


struct StructMemberInfo : public VariableInfo {
    uint32_t offset = 0;
    UniformBufferInfo *uniformBufferInfo = nullptr;

    virtual void setUniformBufferInfo(UniformBufferInfo *uniformBufferInfo);
    template <typename T> StructMemberInfo& operator=(const T& value);
};


struct StructBlockInfo : StructMemberInfo {
    std::vector<StructMemberInfo*> memberInfoList;

    void setUniformBufferInfo(UniformBufferInfo *uniformBufferInfo) override;
    bool isMemberExist(const std::string& memberName) const;
    StructMemberInfo* getMember(const std::string& memberName);
    const StructMemberInfo* getMember(const std::string& memberName) const;
    StructMemberInfo& operator[](const std::string& memberName);
    const StructMemberInfo& operator[](const std::string& memberName) const;
};


struct UniformBufferInfo : StructBlockInfo {
    uint32_t binding = 0;
    uint8_t *buffer = nullptr;

    void setUniformBufferInfo(UniformBufferInfo *uniformBufferInfo) override;
};


template <typename T>
StructMemberInfo& StructMemberInfo::operator=(const T& value)
{
    SYRINX_EXPECT(uniformBufferInfo);
    auto reflType = getReflectionTypeFromCppType<T>();
    if (!(type.basetype == reflType.basetype && type.vecsize == reflType.vecsize && type.columns == reflType.columns)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to update variable [{}] in uniform buffer [{}]",
                                   name, uniformBufferInfo->name);
    }

    auto buffer = uniformBufferInfo->buffer;
    SYRINX_ASSERT(buffer);
    if (offset + sizeof(T) > uniformBufferInfo->size) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to update variable [{}] in uniform buffer [{}]",
            name, uniformBufferInfo->name);
    }

    auto dst = buffer + offset;
    std::memcpy(dst, &value, sizeof(T));
    return *this;
}



class ProgramReflector {
public:
    TextureType getTextureTypeFromReflectionType(const ReflectionType& reflectionType);

public:
    explicit ProgramReflector(const std::string& programName, std::vector<uint32_t>&& sources);
    const std::string& getProgramName() const;
    const std::vector<InterfaceInfo*>& getInputInterfaceList() const;
    const std::vector<InterfaceInfo*>& getOuputInterfaceList() const;
    const std::vector<TextureInfo*>& getSampledTextureList() const;
    const std::vector<UniformBufferInfo*> getUniformBufferList() const;

private:
    void reflectInterfaceList();
    void reflectSampledTextureList();
    void reflectUniformBufferList();
    std::vector<StructMemberInfo*> reflectMemberList(const StructBlockInfo* variable);
    template <typename T> T* create();

private:
    const std::string mProgramName;
    spirv_cross::CompilerGLSL mReflector;
    spirv_cross::ShaderResources mShaderResources;
    spirv_cross::ShaderResources mActiveShaderResource;
    std::vector<InterfaceInfo*> mInputInterfaceList;
    std::vector<InterfaceInfo*> mOutputInterfaceList;
    std::vector<TextureInfo*> mSampledTextureList;
    std::vector<UniformBufferInfo*> mUniformBufferList;
    std::vector<std::unique_ptr<VariableInfo>> mVariableList;
};


template <typename T>
T* ProgramReflector::create()
{
    auto variable = new T();
    mVariableList.push_back(std::unique_ptr<T>(variable));
    return variable;
}

} // namespace Syrinx