#pragma once
#include <string>
#include <functional>
#include <fmt/format.h>
#include <better-enums/enum.h>
#include "Common/SyrinxMacro.h"
#include "Common/SyrinxSingleton.h"
#include "Common/SyrinxPlatform.h"
#include "Time/SyrinxDateTime.h"

#define SYRINX_LOG(message, logLevel) Syrinx::SyrinxLog(message, __CODE__SITE__, logLevel)

#define SYRINX_TRACE(message) SYRINX_LOG(message, Syrinx::ELogLevel::TRACE_LEVEL)
#define SYRINX_DEBUG(message) SYRINX_LOG(message, Syrinx::ELogLevel::DEBUG_LEVEL)
#define SYRINX_INFO(message)  SYRINX_LOG(message, Syrinx::ELogLevel::INFO_LEVEL)
#define SYRINX_WARN(message)  SYRINX_LOG(message, Syrinx::ELogLevel::WARN_LEVEL)
#define SYRINX_ERROR(message) SYRINX_LOG(message, Syrinx::ELogLevel::ERROR_LEVEL)
#define SYRINX_FAULT(message) SYRINX_LOG(message, Syrinx::ELogLevel::FAULT_LEVEL)

#define SYRINX_TRACE_FMT(fmtStr, ...) SYRINX_TRACE(fmt::format(fmtStr, __VA_ARGS__))
#define SYRINX_DEBUG_FMT(fmtStr, ...) SYRINX_DEBUG(fmt::format(fmtStr, __VA_ARGS__))
#define SYRINX_INFO_FMT(fmtStr, ...)  SYRINX_INFO(fmt::format(fmtStr, __VA_ARGS__))
#define SYRINX_WARN_FMT(fmtStr, ...)  SYRINX_WARN(fmt::format(fmtStr, __VA_ARGS__))
#define SYRINX_ERROR_FMT(fmtStr, ...) SYRINX_ERROR(fmt::format(fmtStr, __VA_ARGS__))
#define SYRINX_FAULT_FMT(fmtStr, ...) SYRINX_FAULT(fmt::format(fmtStr, __VA_ARGS__))


namespace Syrinx {

BETTER_ENUM(ELogLevel, std::uint8_t, TRACE_LEVEL, DEBUG_LEVEL, INFO_LEVEL, WARN_LEVEL, ERROR_LEVEL, FAULT_LEVEL, NUM_LOG_LEVEL);

extern void SyrinxLog(const std::string& message, const std::string& fileName, const std::string& functionName, int lineNumber, ELogLevel logLevel);

class LogSource {
public:
	LogSource(const std::string& fileName, const std::string& functionName, int lineNumber);

public:
	const std::string fileName;
	const std::string functionName;
	const int lineNumber;
};

class LogInfo {
public:
	LogInfo(const std::string& message, const LogSource& source, const DateTime& time, ELogLevel logLevel);

public:
	const std::string message;
	const LogSource source;
	const DateTime time;
	const ELogLevel logLevel;
};


std::string DefaultMessageFormater(const LogInfo& log);


class OutputStream {
public:
	virtual void add(const std::string& message);
};


class LogManager : public Singleton<LogManager> {
public:
	using MessageFormater = std::function<std::string(const LogInfo&)>;

public:
	LogManager(const MessageFormater& formater = DefaultMessageFormater, const OutputStream& outputStream = OutputStream());
	~LogManager() = default;
	LogManager(const LogManager&) = delete;
	LogManager& operator=(const LogManager&) = delete;

	void add(const LogInfo& log);

private:
	MessageFormater mFormater;
	OutputStream mOutputStream;
};

} // namespace Syrinx