#include "Logging/SyrinxLogManager.h"
#include <iostream>

namespace Syrinx {

std::string DefaultMessageFormater(const LogInfo& log)
{
	const std::string formatString =
		"Level: {logLevel}\n"
		"Time: {year}/{month}/{day} {hour}:{minute}:{second}.{millisecond}\n"
		"Source: file={fileName} function={functionName} line number={lineNumber}\n"
		"Message: {message}\n";

	return fmt::format(formatString,
					   fmt::arg("logLevel", log.logLevel._to_string()),
					   fmt::arg("message", log.message),
					   fmt::arg("year", log.time.getYear()),
					   fmt::arg("month", log.time.getMonth()),
					   fmt::arg("day", log.time.getDay()),
					   fmt::arg("hour", log.time.getHour()),
					   fmt::arg("minute", log.time.getMinute()),
					   fmt::arg("second", log.time.getSecond()),
					   fmt::arg("millisecond", log.time.getMillisecond()),
					   fmt::arg("fileName", log.source.fileName),
					   fmt::arg("functionName", log.source.functionName),
					   fmt::arg("lineNumber", log.source.lineNumber));
}


void SyrinxLog(const std::string& message, const std::string& fileName, const std::string& functionName, int lineNumber, ELogLevel logLevel)
{
    SYRINX_EXPECT(!message.empty() && !fileName.empty() && !functionName.empty());

	DateTime time = DateTime::now();
	LogSource source{ fileName, functionName, lineNumber };
	LogInfo log{ message, source, time, logLevel };
	LogManager::getInstance().add(log);
}


LogSource::LogSource(const std::string& fileName, const std::string& functionName, int lineNumber)
	: fileName(fileName), functionName(functionName), lineNumber(lineNumber)
{
	SYRINX_EXPECT(!fileName.empty() && !functionName.empty() && lineNumber >= 0);
}


LogInfo::LogInfo(const std::string& message, const LogSource& source, const DateTime& time, ELogLevel logLevel)
	: message(message), source(source), time(time), logLevel(logLevel)
{
	SYRINX_EXPECT(!message.empty());
}


void OutputStream::add(const std::string& message)
{
	std::cout << message << std::endl;
}


LogManager::LogManager(const MessageFormater& formater, const OutputStream& outputStream)
	: mFormater(formater), mOutputStream(outputStream)
{
    SYRINX_EXPECT(formater);
}

void LogManager::add(const LogInfo& log)
{
	std::string logString = mFormater(log);
	mOutputStream.add(logString);
}

} // namespace Syrinx