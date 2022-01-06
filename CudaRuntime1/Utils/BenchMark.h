#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <algorithm>

struct ProfileResult
{
	std::string Name;
	long long Start, End;
};
struct InstrumentationSession
{
	std::string Name;
	InstrumentationSession(const std::string name) :Name(name) { }
};

class Instrumentor {
private:
	InstrumentationSession* m_CurrentSession;
	std::ofstream m_outputStream;
	int m_ProfileCount;
public:
	Instrumentor()
		:m_CurrentSession(nullptr), m_ProfileCount(0) {

	}

	void BeginSession(const std::string& name, const std::string& filepath = "results.json") {
		m_outputStream.open(filepath);
		WriteHeader();
		m_CurrentSession = new InstrumentationSession(name);
	}

	void EndSession() {
		WritFooter();
		m_outputStream.close();
		delete m_CurrentSession;
		m_CurrentSession = nullptr;
		m_ProfileCount = 0;
	}

	void WriteProfile(const ProfileResult& result) {
		if (m_ProfileCount++ > 0) {
			m_outputStream << " ,";
		}
		std::string name = result.Name;
		std::replace(name.begin(), name.end(), '"', '\'');
		m_outputStream << "{";
		m_outputStream << "\"cat\":\"function\",";
		m_outputStream << "\"dur\":" << (result.End - result.Start) << ",";
		m_outputStream << "\"name\":\"" << name << "\",";
		m_outputStream << "\"ph\":\"X\",";
		m_outputStream << "\"pid\":0,";
		m_outputStream << "\"tid\":0,";
		m_outputStream << "\"ts\":" << result.Start;
		m_outputStream << "}";

		m_outputStream.flush();
	}
	void WriteHeader() {
		m_outputStream << "{\"otherData\":{},\"traceEvents\":[";
		m_outputStream.flush();
	}
	void WritFooter() {
		m_outputStream << "]}";
		m_outputStream.flush();
	}
	static Instrumentor& Get() {
		static Instrumentor* instance = new Instrumentor();
		return *instance;
	}
};

class InstrumentationTimer
{
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
	const char* m_Name;
	bool m_Stopped;
public:
	InstrumentationTimer(const char* name) :
		m_Name(name), m_Stopped(false) {
		m_StartTimepoint = std::chrono::high_resolution_clock::now();
	}
	~InstrumentationTimer() {
		if (!m_Stopped)
			Stop();
	}
	void Stop() {
		auto endTimepoint = std::chrono::high_resolution_clock::now();
		auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
		auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

		auto duration = end - start;

		Instrumentor::Get().WriteProfile({ m_Name, start, end });
		std::cout << m_Name << ": " << duration << " us ("<< duration / 1000.0 << "ms)\n";
		m_Stopped = true;
	}
};