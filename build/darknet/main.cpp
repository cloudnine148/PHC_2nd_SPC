#include "DetectionProcessing.hpp"

int main(int argc, char* argv[])
{
	DetectionProcessing dp = DetectionProcessing(argv, argc);
	dp.mainProcessing(argc, argv);
	dp.~DetectionProcessing();
}