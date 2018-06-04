#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

vector<vector<double>> parseSeeds()
{
    ifstream in("seeds.input");
    vector<vector<double>> fields;

    if (in) {
        string line;

        while (getline(in, line)) {
            stringstream sep(line);
            string field;

            fields.push_back(vector<double>());

            while (getline(sep, field, ',')) {
                fields.back().push_back(stod(field));
            }
        }
    }
    /*
    for (auto row : fields) {
        for (auto field : row) {
            cout << field << " ";
        }

        cout << '\n';
    }
    */
    return fields;
}

int main() {
    vector<vector<double>> fields = parseSeeds();

    for (auto row : fields) {
        for (auto field : row) {
            cout << field << " ";
        }

        cout << '\n';
    }
}
