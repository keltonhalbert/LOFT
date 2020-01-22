#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

int main() {
    ifstream cFile("parcel.namelist");
    if (cFile.is_open()) {
        string line;
        while(getline(cFile, line)) {
            line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
            if (line[0] == '#' || line.empty()) continue;
            auto delimiterPos = line.find("=");
            auto name = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);
            cout << name << " " << value << endl;
        }
    }
    else {
        cerr << "Couldn't open namelist file for reading." << endl;
    }
}
