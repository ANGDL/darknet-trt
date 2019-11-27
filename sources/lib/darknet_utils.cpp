#include "darknet_utils.h"

bool file_exits(const std::string filename)
{
	return std::filesystem::exists(std::filesystem::path(filename));
}

static void leftTrim(std::string& s)
{
	s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
	s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
	leftTrim(s);
	rightTrim(s);
	return s;
}
