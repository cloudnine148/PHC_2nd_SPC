#pragma once
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <mutex>/
#include <condition_variable>

#include "opencv2/opencv.hpp"
#include "opencv2/videoio/videoio.hpp"

#include "yolo_v2_class.hpp"
#include "DataCommunication.hpp"
#include "Validation.hpp"
#include "common.hpp"

#define FOCAL_LENGTH	500
#define REAL_PEOPLE_HEIGHT 1.68
#define MAX_DISTANCE 35.0

#define SAVE_CHECK true 

#define RESULT_DISPLAY		1	// 1 : Display, 0 : No Display
#define DEBUG_MODE			0
enum Orientation{RIGHT,RIGHTFRONT,FRONT,LEFTFRONT,LEFT,LEFTBACK,BACK,RIGHTBACK };
int det_compare(const void* m, const void *n);

struct DetPeople_INFO
{
	int start_x;
	int end_x;
	int start_y;
	int end_y;
	int obj_class;
	int det_id;

	int nSuddenCrossing;				// 갑자기 뛰어드는 보행자 유무 (0:초기화, -1: false detection, 1: suddenCrossing
	double dProbWarning;				// 갑자기 뛰어드는 보행자 위험 정도
	double distance;					//  차량과 보행자간의 거리값 

	int _prt_position;					// 0:보행자 검출 없음, 1:Center, 2:Left, 3:Right, 4:판단 보류
	int _prt_risk;						// 0:보행자 검출 없음, 1:Normal, 2:Caution, 3:Warning
	int _prt_direction;					// 0:보행자 검출 없음, 1:정지, 2:왼쪽->오른쪽, 3:오른쪽->왼쪽, 4:판단보류
	int _prt_distance;					// 0:보행자 검출 없음, 1:5m, 2:10m, 3:15m, 4:20m, 5:25m, 6:30m, 7:35m, 8:40m 9:40m이상
	int _prt_riskRate;					// 보행자 위험도 스코어 값 = 실제값 * 100
	DetPeople_INFO()
	{
		nSuddenCrossing = 0;
		dProbWarning = 0.0;
	}
};


class DetectionProcessing
{

private:
	std::string filename;
	std::string cfgName;
	std::string weightName;
	std::vector<std::string> obj_names;
	std::string out_videofile;
	std::string select_mode;
	bool bFlag = false;
	std::string current_file_name;
	bool save_output_videofile;

	// 현재 프레임 번호 
	int m_nCurrentFrameNum;
	// 이전 프레임 번호	
	int m_nPrevFrameNum;
	// 프레임 카운트
	int m_nFrameCount;
	double avg_time = 0;

	//0 -> Day
	//1 -> Night 
	int day_or_night = 0;
	// Reference Line을 위한 Mask 설정
	BYTE** m_mask;					// Reference Line의 Mask
	BYTE** matImg_ptr;				// GrayScale MatImg 를 BYTE형 주소 저장
	IplImage* m_ref_mask_img;

	CvPoint referenceLine_ptss[4];
									// 자동차 주행 방향
	bool m_bUpdateMask;				// 현재 프레임의 마스크 업데이트 유무


public:
	DataCommunication *dataCommunication_ptr;
	DetectionProcessing(char *argv[], int argc);
	~DetectionProcessing();
	void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
		unsigned int wait_msec = 0, int current_det_fps = -1, int current_cap_fps = -1);
	void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names);
	std::vector<std::string> objects_names_from_file(std::string const filename);
	int mainProcessing(int argc, char *argv[]);
	void valueInit();
	void valueDestroy();
	int OpticalFlowProcessing(cv::Mat current_Img, int nCurrentFrame);
	void Display_referenceMask(cv::Mat current_Img);
	void ExtractSuddenCrossingFeatureInCropedObject();
	void detectionSuddenCrossingPedestrian(cv::Mat currImg, std::vector<bbox_t> result_vec);
	double detectionSuddenCrossing_overlappedAreaRate(int start_x, int end_x, int start_y, int end_y,
		int& nWindowPosition);
	void DetectionProcessing::detectionSuddenCrossing_movement(cv::Mat currImg, int nDirIndex, 
		int nWindowPositionIn3part, double nDistanceFromCam, double& dResultDir, double& dResultDistance);

	double detectionGaussianProbabilityDensityFunction(double value, double mean, double standardDeviation);
	int extractDistanceBetweenWinNCam(int nWindowHeight);
	void Display_StaticReferenceMask(cv::Mat currImg, int day_or_night);
	void convertDataToProtocolMsg(DetPeople_INFO *detInfo, int detCnt, int day_night);
	void draw_boxes_validation(cv::Mat mat_img, truth_box* all_truth_box, std::vector<bbox_t> result_vec,
		int num_labels, FILE *fp, int frame_no, int *current_TP, int *current_FP);
	void detection_sudden_crossing_Validation(cv::Mat currImg, std::vector<bbox_t> result_vec,
		det_box *all_det_boxes);
	void Display_StaticReferenceMask_Validation(cv::Mat currImg);
	void set_StaticReferenceMask();
	void find_replace(char* str, char* orig, char* rep, char* output);
	void convert_gt_coordinate(cv::Mat currImg, truth_box* ground_truth, int num_labels);
	void detection_tp_check(truth_box* t_box, det_box* d_box, tp_box* tp_box, int num_labels,
		int all_det_cnt, int *tp_cnt);
	void draw_box_test(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names);
	void orientation_display(cv::Mat curr_img, int ped_orientation, DetPeople_INFO det_info);
	int decide_day_and_night(cv::Mat mat_img);
	void get_ReferenceLine_Mask();
};
