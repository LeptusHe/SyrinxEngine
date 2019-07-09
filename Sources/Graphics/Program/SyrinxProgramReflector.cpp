#include "SyrinxProgramReflector.h"
#include <Exception/SyrinxException.h>

namespace Syrinx {

TextureType ProgramReflector::getTextureTypeFromReflectionType(const ReflectionType& reflectionType)
{
    SYRINX_EXPECT(reflectionType.basetype == spirv_cross::SPIRType::BaseType::SampledImage);
    SYRINX_EXPECT(reflectionType.image.sampled == 1);

#define ref_to_tex(dim_type, tex_type) if (imageInfo.dim == dim_type) return tex_type

    const auto& imageInfo = reflectionType.image;
    if (!imageInfo.arrayed) {
        ref_to_tex(spv::Dim2D, TextureType::TEXTURE_2D);
        ref_to_tex(spv::Dim3D, TextureType::TEXTURE_3D);
        ref_to_tex(spv::DimCube, TextureType::TEXTURE_CUBEMAP);
    } else {
        ref_to_tex(spv::Dim2D, TextureType::TEXTURE_2D_ARRAY);
    }
#undef ref_to_tex

    SHOULD_NOT_GET_HERE();
    return TextureType::UNDEFINED;
}


ProgramReflector::ProgramReflector(const std::string& programName, std::vector<uint32_t>&& sources)
    : mProgramName(programName)
    , mReflector(std::move(sources))
{
    SYRINX_ENSURE(!mProgramName.empty());
    mShaderResources = mReflector.get_shader_resources();
    auto activeInterfaceVariables = mReflector.get_active_interface_variables();
    mActiveShaderResource = mReflector.get_shader_resources(activeInterfaceVariables);
    reflectInterfaceList();
    reflectSampledTextureList();
    reflectUniformBufferList();
}


void ProgramReflector::reflectInterfaceList()
{
    for (const auto& input : mActiveShaderResource.stage_inputs) {
        auto inputInterface = create<InterfaceInfo>();
        inputInterface->name = input.name;
        inputInterface->type = mReflector.get_type(input.base_type_id);
        inputInterface->location = mReflector.get_decoration(input.id, spv::DecorationLocation);
        mInputInterfaceList.push_back(inputInterface);
    }

    for (const auto& output : mActiveShaderResource.stage_outputs) {
        auto outputInterface = create<InterfaceInfo>();
        outputInterface->name = output.name;
        outputInterface->type = mReflector.get_type(output.base_type_id);
        outputInterface->location = mReflector.get_decoration(output.id, spv::DecorationLocation);
    }
}


void ProgramReflector::reflectSampledTextureList()
{
    for (const auto& sampledTexture : mActiveShaderResource.sampled_images) {
        auto sampledTextureInfo = create<TextureInfo>();
        sampledTextureInfo->name = sampledTexture.name;
        auto reflectionType = mReflector.get_type(sampledTexture.base_type_id);
        sampledTextureInfo->type = getTextureTypeFromReflectionType(reflectionType);
        sampledTextureInfo->size = sizeof(uint64_t);
        sampledTextureInfo->binding = mReflector.get_decoration(sampledTexture.id, spv::DecorationBinding);
        mSampledTextureList.push_back(sampledTextureInfo);
    }
}


void ProgramReflector::reflectUniformBufferList()
{
    for (const auto& uniformBuffer : mActiveShaderResource.uniform_buffers) {
        auto uniformBufferInfo = create<UniformBufferInfo>();
        const auto type = mReflector.get_type(uniformBuffer.base_type_id);

        uniformBufferInfo->name = uniformBuffer.name;
        uniformBufferInfo->type = type;
        uniformBufferInfo->size = mReflector.get_declared_struct_size(type);
        uniformBufferInfo->offset = 0;
        uniformBufferInfo->binding = mReflector.get_decoration(uniformBuffer.id, spv::DecorationBinding);
        uniformBufferInfo->uniformBufferInfo = uniformBufferInfo;
        SYRINX_ASSERT(uniformBufferInfo->uniformBufferInfo == uniformBufferInfo);
        SYRINX_ASSERT(!uniformBufferInfo->buffer);
        uniformBufferInfo->memberInfoList = reflectMemberList(uniformBufferInfo);
        mUniformBufferList.push_back(uniformBufferInfo);
    }
}


const std::string& ProgramReflector::getProgramName() const
{
    return mProgramName;
}


const std::vector<InterfaceInfo*>& ProgramReflector::getInputInterfaceList() const
{
    return mInputInterfaceList;
}


const std::vector<InterfaceInfo*>& ProgramReflector::getOuputInterfaceList() const
{
    return mOutputInterfaceList;
}


const std::vector<TextureInfo*>& ProgramReflector::getSampledTextureList() const
{
    return mSampledTextureList;
}


const std::vector<UniformBufferInfo*> ProgramReflector::getUniformBufferList() const
{
    return mUniformBufferList;
}


std::vector<StructMemberInfo*> ProgramReflector::reflectMemberList(const StructBlockInfo *structBlockInfo)
{
    SYRINX_EXPECT(structBlockInfo);
    uint32_t offset = structBlockInfo->offset;
    auto structBlockType = structBlockInfo->type;
    auto memberCount = structBlockType.member_types.size();

    std::vector<StructMemberInfo*> memberInfoList;
    for (int i = 0; i < memberCount; ++ i) {
        const std::string memberName = mReflector.get_member_name(structBlockType.self, i);
        auto& memberType = mReflector.get_type(structBlockType.member_types[i]);
        auto memberSize = mReflector.get_declared_struct_member_size(structBlockType, i);
        auto memberOffset = mReflector.type_struct_member_offset(structBlockType, i);

        auto parseVariableInfo = [&, structBlockInfo](StructMemberInfo *member) {
            member->name = memberName;
            member->type = memberType;
            member->size = memberSize;
            member->offset = memberOffset + offset;
            member->uniformBufferInfo = structBlockInfo->uniformBufferInfo;
        };

        StructMemberInfo *structMember = nullptr;
        if (memberType.basetype == ReflectionType::Struct) {
            auto member = create<StructBlockInfo>();
            parseVariableInfo(member);
            member->memberInfoList = std::move(reflectMemberList(member));
            structMember = member;
        } else {
            auto member = create<StructMemberInfo>();
            parseVariableInfo(member);
            structMember = member;
        }
        SYRINX_ASSERT(structMember);
        memberInfoList.push_back(structMember);
    }
    return memberInfoList;
}


bool StructBlockInfo::isMemberExist(const std::string& memberName) const
{
    for (const auto& memberInfo : memberInfoList) {
        if (memberInfo->name == memberName) {
            return true;
        }
    }
    return false;
}


const StructMemberInfo* StructBlockInfo::getMember(const std::string& memberName) const
{
    for (const auto memberInfo : memberInfoList) {
        if (memberInfo->name == memberName) {
            return memberInfo;
        }
    }
    return nullptr;
}


StructMemberInfo* StructBlockInfo::getMember(const std::string& memberName)
{
    for (const auto memberInfo : memberInfoList) {
        if (memberInfo->name == memberName) {
            return memberInfo;
        }
    }
    return nullptr;
}


const StructMemberInfo& StructBlockInfo::operator[](const std::string& memberName) const
{
    auto memberInfo = getMember(memberName);
    if (!memberInfo) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to get member [{}] for [{}]", memberName, name);
    }
    return *memberInfo;
}


StructMemberInfo& StructBlockInfo::operator[](const std::string& memberName)
{
    auto memberInfo = getMember(memberName);
    if (!memberInfo) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to get member [{}] for [{}]", memberName, name);
    }
    return *memberInfo;
}



void StructMemberInfo::setUniformBufferInfo(UniformBufferInfo *uboInfo)
{
    uniformBufferInfo = uboInfo;
}


void StructBlockInfo::setUniformBufferInfo(UniformBufferInfo *uboInfo)
{
    uniformBufferInfo = uboInfo;
    for (auto& memberInfo : memberInfoList) {
        memberInfo->setUniformBufferInfo(uboInfo);
    }
}


void UniformBufferInfo::setUniformBufferInfo(UniformBufferInfo *uboInfo)
{
    uniformBufferInfo = uboInfo;
    for (auto& memberInfo : memberInfoList) {
        memberInfo->setUniformBufferInfo(uboInfo);
    }
}

} // namespace Syrinx
