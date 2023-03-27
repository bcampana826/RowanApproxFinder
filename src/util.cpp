#include "../inc/util.h"



unsigned long long split(const string &txt, vector<string> &strs, char ch) {
	//this is the general case
	size_t pos = txt.find(ch);
	size_t initialPos = 0;
	strs.clear();
	// Decompose statement
	while (pos != string::npos) {
		strs.push_back(txt.substr(initialPos, pos - initialPos + 1));
		initialPos = pos + 1;
		pos = txt.find(ch, initialPos);
	}
	// Add the last one
	strs.push_back(txt.substr(initialPos, min(pos, txt.size()) - initialPos + 1));
	//return the size of the vector
	return strs.size();
}