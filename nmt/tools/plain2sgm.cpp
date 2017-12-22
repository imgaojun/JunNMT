#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace std;

void OutSingleFile(const vector<string>& ifile, const string& ofile, const string& dir = "src")
{
	//ofstream fout(ofile.c_str());
	if (dir == "src")
		cout << "<srcset setid=\"set\" srclang=\"src\" trglang=\"tgt\">" << endl;
	else if (dir == "ref")
		cout << "<refset setid=\"set\" srclang=\"src\" trglang=\"tgt\">" << endl;
	else if (dir == "tst")
		cout << "<tstset setid=\"set\" srclang=\"src\" trglang=\"tgt\">" << endl;
	
	for (unsigned int i = 0; i < ifile.size(); i++)
	{
		if (dir == "ref")
			cout << "<DOC docid=\"doc\" sysid=\"" << dir << i << "\">" << endl;
		else
			cout << "<DOC docid=\"doc\" sysid=\"" << dir << "\">" << endl;
		ifstream fin(ifile[i].c_str());
		string line;
		unsigned int line_num = 1;
		while (getline(fin, line))
		{
			cout << "<seg id=\"" << line_num++ << "\">" << line << "</seg>" << endl;
		}
		cout << "</DOC>" << endl;
	}

	if (dir == "ref")
		cout << "</refset>" << endl;
	else if (dir == "src")
		cout << "</srcset>" << endl;
	else if (dir == "tst")
		cout << "</tstset>" << endl;
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cerr << "[src|ref|tst][plain0][...][plainN]" << endl;
        cerr << "Example :" << endl;
        cerr << "./plain2sgm src src.plain > src.sgm" << endl;
        cerr << "./plain2sgm ref ref.plain0 ref.plain1 ref.plain2 .... > ref.sgm" << endl;
        cerr << "./plain2sgm tst tst.plain > tst.sgm" << endl;
		exit(1);
	}
	if (string(argv[1]) != "src" && string(argv[1]) != "ref" && string(argv[1]) != "tst")
	{
		cerr << "[src|ref|tst][plain0][...][plainN]" << endl;
		exit(1);
	}
	if (string(argv[1]) != "ref" && argc > 3)
	{
		cerr << "Error: only refset can get multi plain !" << endl;
		exit(1);
	}
	vector<string> vFile;
	string ofile(argv[2]);
	ofile += ".sgm";
	for (unsigned int i = 2; i < argc; i++)
	{
		vFile.push_back(argv[i]);
	}
	OutSingleFile(vFile, ofile, argv[1]);
	return 0;
}
