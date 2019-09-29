#pragma once
#include <string>
#include <pugixml.hpp>
#include <Exception/SyrinxException.h>

namespace Syrinx {

inline bool hasChild(const pugi::xml_node& node, const std::string& childName)
{
    auto childNode = node.child(childName.c_str());
    return childNode.empty();
}


inline pugi::xml_node getChild(const pugi::xml_node& node, const std::string& childName)
{
    auto childNode = node.child(childName.c_str());
    if (childNode.empty()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "node [{}] does not have child [{}]", node.name(), childName);
    }
    return childNode;
}


inline pugi::xml_attribute getAttribute(const pugi::xml_node& node, const std::string& attributeName)
{
    auto attribute = node.attribute(attributeName.c_str());
    if (attribute.empty()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "node [{}] does not have attribute [{}]", node.name(), attributeName);
    }
    return attribute;
}


inline std::string getText(const pugi::xml_node& node)
{
    auto text = node.text();
    if (text.empty()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "node [{}] does not have text", node.name());
    }
    return text.as_string();
}

} // namespace Syrinx