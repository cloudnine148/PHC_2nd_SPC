#include "DetectionProcessing.hpp"
#include <iterator>
#include <time.h>
#include <algorithm>
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <direct.h>
#include <io.h>
#ifdef _DEBUG
#define malloc(s) _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#if defined(__cplusplus)
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif
#endif
static double normal_gt_count = 0;
static double normal_st_count = 0;
static double caution_gt_count = 0;
static double caution_st_count = 0;
static double warning_gt_count = 0;
static double warning_st_count = 0;
bool cmp_Distance(DetPeople_INFO a, DetPeople_INFO b) { return a.distance < b.distance; }

class extrapolate_coords_t {
public:
	std::vector<bbox_t> old_result_vec;
	std::vector<float> dx_vec, dy_vec, time_vec;
	std::vector<float> old_dx_vec, old_dy_vec;

	void new_result(std::vector<bbox_t> new_result_vec, float new_time) {
		old_dx_vec = dx_vec;
		old_dy_vec = dy_vec;
		if (old_dx_vec.size() != old_result_vec.size()) std::cout << "old_dx != old_res \n";
		dx_vec = std::vector<float>(new_result_vec.size(), 0);
		dy_vec = std::vector<float>(new_result_vec.size(), 0);
		update_result(new_result_vec, new_time, false);
		old_result_vec = new_result_vec;
		time_vec = std::vector<float>(new_result_vec.size(), new_time);
	}

	void update_result(std::vector<bbox_t> new_result_vec, float new_time, bool update = true) {
		for (size_t i = 0; i < new_result_vec.size(); ++i) {
			for (size_t k = 0; k < old_result_vec.size(); ++k) {
				if (old_result_vec[k].track_id == new_result_vec[i].track_id && old_result_vec[k].obj_id == new_result_vec[i].obj_id) {
					float const delta_time = new_time - time_vec[k];
					if (abs(delta_time) < 1) break;
					size_t index = (update) ? k : i;
					float dx = ((float)new_result_vec[i].x - (float)old_result_vec[k].x) / delta_time;
					float dy = ((float)new_result_vec[i].y - (float)old_result_vec[k].y) / delta_time;
					float old_dx = dx, old_dy = dy;

					// if it's shaking
					if (update) {
						if (dx * dx_vec[i] < 0) dx = dx / 2;
						if (dy * dy_vec[i] < 0) dy = dy / 2;
					}
					else {
						if (dx * old_dx_vec[k] < 0) dx = dx / 2;
						if (dy * old_dy_vec[k] < 0) dy = dy / 2;
					}
					dx_vec[index] = dx;
					dy_vec[index] = dy;

					//if (old_dx == dx && old_dy == dy) std::cout << "not shakin \n";
					//else std::cout << "shakin \n";

					if (dx_vec[index] > 1000 || dy_vec[index] > 1000) {
						//std::cout << "!!! bad dx or dy, dx = " << dx_vec[index] << ", dy = " << dy_vec[index] <<
						//    ", delta_time = " << delta_time << ", update = " << update << std::endl;
						dx_vec[index] = 0;
						dy_vec[index] = 0;
					}
					old_result_vec[k].x = new_result_vec[i].x;
					old_result_vec[k].y = new_result_vec[i].y;
					time_vec[k] = new_time;
					break;
				}
			}
		}
	}

	std::vector<bbox_t> predict(float cur_time) {
		std::vector<bbox_t> result_vec = old_result_vec;
		for (size_t i = 0; i < old_result_vec.size(); ++i) {
			float const delta_time = cur_time - time_vec[i];
			auto &bbox = result_vec[i];
			float new_x = (float)bbox.x + dx_vec[i] * delta_time;
			float new_y = (float)bbox.y + dy_vec[i] * delta_time;
			if (new_x > 0) bbox.x = new_x;
			else bbox.x = 0;
			if (new_y > 0) bbox.y = new_y;
			else bbox.y = 0;
		}
		return result_vec;
	}
};
DetectionProcessing::DetectionProcessing(char *argv[], int argc)
{
#if DEBUG_MODE == 1
	cfgName = CFG_NAME;
	weightName = WEIGHT_NAME;
	obj_names = objects_names_from_file(OBJECT_NAME);
	out_videofile = RESULT_FILE;
#elif DEBUG_MODE == 0
	cfgName = argv[2];
	weightName = argv[3];
	obj_names = objects_names_from_file(argv[1]);
	out_videofile = RESULT_FILE;
#endif
	save_output_videofile = SAVE_CHECK;
	dataCommunication_ptr = new DataCommunication(argv[5]);
	valueInit();
	std::cout << "Command Start...." << std::endl;
	//초기 0x01메세지 전달
	dataCommunication_ptr->Command_Start();
}

DetectionProcessing::~DetectionProcessing()
{
	valueDestroy();
	dataCommunication_ptr->Command_Stop();
	dataCommunication_ptr->disconnect();
	delete dataCommunication_ptr;
	exit(0);
}
void DetectionProcessing::valueDestroy()
{
	for (int i = 0; i < IMG_SIZE_HEIGHT; i++)
	{
		delete[] m_mask[i];
		delete[] matImg_ptr[i];
	}
	delete[] m_mask;
	delete[] matImg_ptr;
	cvReleaseImage(&m_ref_mask_img);
}

void DetectionProcessing::valueInit()
{
	// Optical Flow 클래스 초기화
	//optFlow.Func_Init(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, OPTICALfLOW_GRID_INTERVAL);
	m_bUpdateMask = 0;
	m_nPrevFrameNum = 99999;
	int nOPFFeatureCount = (IMG_SIZE_HEIGHT / OPTICALfLOW_GRID_INTERVAL + 1) * (IMG_SIZE_WIDTH / OPTICALfLOW_GRID_INTERVAL + 1);
	//m_opticalflow_result = new OPTICALFLOW_FEATURES[nOPFFeatureCount];

	// Reference Line을 위한 Mask 추출

	m_mask = new BYTE*[IMG_SIZE_HEIGHT];
	matImg_ptr = new BYTE*[IMG_SIZE_HEIGHT];

	for (int i = 0; i < IMG_SIZE_HEIGHT; i++)
	{
		m_mask[i] = new BYTE[IMG_SIZE_WIDTH];
		matImg_ptr[i] = new BYTE[IMG_SIZE_WIDTH];
	}
	m_ref_mask_img = cvCreateImage(cvSize(IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT), IPL_DEPTH_8U, 1);
	cvSetZero(m_ref_mask_img);
}

void DetectionProcessing::draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec,
	std::vector<std::string> obj_names, unsigned wait_msec, int current_det_fps, int current_cap_fps)
{
	clock_t begin, end;
	begin = clock();
	int pt_cnt = 0;

	Display_StaticReferenceMask(mat_img, day_or_night);
	set_StaticReferenceMask();

#if RESULT_DISPLAY
	if (day_or_night == 0)
		putText(mat_img, "DayTime", cv::Point2f(IMG_SIZE_WIDTH*0.078, IMG_SIZE_HEIGHT*0.097),
			cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(255, 0, 0), 2);
	else
		putText(mat_img, "NightTime", cv::Point2f(IMG_SIZE_WIDTH*0.078, IMG_SIZE_HEIGHT*0.097),
			cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(120, 250, 255), 2);
#endif
	//if (m_nCurrentFrameNum > 2)
		//갑자기 끼어드는 보행자 검출
		detectionSuddenCrossingPedestrian(mat_img, result_vec);

	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	cv::Point ptss[4];
	cv::Mat ref_Mask(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, CV_8UC1);
	cv::Mat det_People(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, CV_8UC1);
	cv::Mat and_Img(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, CV_8UC1);
	ref_Mask.setTo(cv::Scalar(0));
	det_People.setTo(cv::Scalar(0));
	and_Img.setTo(cv::Scalar(0));
	int cntPixel = 0;
	double overlap_Ratio = 0.0;
	for (auto &i : result_vec) {
		//detection결과가 person일때 만
		//if (i.obj_id == 0)		//coco data
		//if (0 <= i.obj_id && i.obj_id <8)		//coco data
		if (1)
			//if(i.obj_id == 14)			//voc data
		{
			// 검출된 사람의 박스 정보
			int const offset = i.obj_id * 123457 % 6;
			int const color_scale = 150 + (i.obj_id * 123457) % 100;
			cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
			//cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), cv::Scalar(0, 255, 0), 1);

			color *= color_scale;

			////Frame Informaition display
			//if (obj_names.size() > i.obj_id) {
			//	std::string obj_name = obj_names[i.obj_id];
			//if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			//cv::Size const text_size = cv::getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			//int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			//cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 3, 0), std::max((int)i.y - 30, 0)),
			//cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
			//color, CV_FILLED, 8, 0);
			//	putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			//}
		}
	}
	//if (current_det_fps >= 0 && current_cap_fps >= 0) {
		/*std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
		putText(mat_img, fps_str, cv::Point2f(IMG_SIZE_WIDTH * 0.5208, IMG_SIZE_HEIGHT*0.097), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50, 255, 0), 1);*/
		//}
	cv::imshow("SPC Result", mat_img);
	cv::waitKey(3);
	dataCommunication_ptr->Command_Signal();

	/*std::string frameNo_str = "Frame No : " + std::to_string(m_nCurrnetFrameNum);
	putText(mat_img, frameNo_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
	cv::imshow("window name", mat_img);
	cv::waitKey(wait_msec);*/
}

std::vector<std::string> DetectionProcessing::objects_names_from_file(std::string const filename)
{
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for (std::string line; getline(file, line);) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}

void DetectionProcessing::show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names)
{
	for (auto &i : result_vec) {
		if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
		std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
			<< ", w = " << i.w << ", h = " << i.h
			<< std::setprecision(3) << ", prob = " << i.prob << std::endl;
	}
}

int DetectionProcessing::mainProcessing(int argc, char *argv[])
{
	m_nCurrentFrameNum = 0;
	m_nFrameCount = 0;
#if DEBUG_MODE  == 1
	//Debugging
	filename = "C:\\Users\\USER\\Desktop\\PHC_Detection_Ve0.3 - Release\\build\\darknet\\x64\\video2.mp4";
	//filename = "C:\\Users\\USER\\Desktop\\Video_003.avi";
#elif DEBUG_MODE == 0
	//Testing
	if (argc > 1) filename = argv[4];
	/*std::string names_file = "data/yolo.names";
	std::string cfg_file = "yolov3-tiny.cfg";
	std::string weights_file = "yolov3-tiny_final.weights";*/

	if (argc > 4) {	//voc.names yolo-voc.cfg yolo-voc.weights test.mp4
		obj_names = objects_names_from_file(argv[1]);
		cfgName = argv[2];
		weightName = argv[3];
		filename = argv[4];
		//select_mode = argv[6];
	}

	//float const thresh = (argc > 5) ? std::stof(argv[5]) : 0.20;
	float const thresh = 0.20;

#endif

	Detector detector(cfgName, weightName);

	while (true)
	{
		std::cout << "input image or video filename: ";
		if (filename.size() == 0) std::cin >> filename;
		if (filename.size() == 0) break;

		try {
#ifdef OPENCV
			extrapolate_coords_t extrapolate_coords;
			bool extrapolate_flag = false;
			float cur_time_extrapolate = 0, old_time_extrapolate = 0;
			preview_boxes_t large_preview(100, 150, false), small_preview(50, 50, true);
			bool show_small_boxes = false;

			std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);
			std::string const protocol = filename.substr(0, 7);
			if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "mov" ||     // video file
				protocol == "rtmp://" || protocol == "rtsp://" || protocol == "http://" || protocol == "https:/")    // video network stream
			{
				cv::Mat cap_frame, cur_frame, det_frame, write_frame;
				int passed_flow_frames = 0;
				std::shared_ptr<image_t> det_image;
				std::vector<bbox_t> result_vec, thread_result_vec;
				detector.nms = 0.02;    // comment it - if track_id is not required
				std::atomic<bool> consumed, videowrite_ready;
				bool exit_flag = false;
				consumed = true;
				videowrite_ready = true;
				std::atomic<int> fps_det_counter, fps_cap_counter;
				fps_det_counter = 0;
				fps_cap_counter = 0;
				int current_det_fps = 0, current_cap_fps = 0;
				std::thread t_detect, t_cap, t_videowrite;
				std::mutex mtx;
				std::condition_variable cv_detected, cv_pre_tracked;
				std::chrono::steady_clock::time_point steady_start, steady_end;
				
				cv::VideoCapture cap(filename); cap >> cur_frame;

				int const video_fps = cap.get(CV_CAP_PROP_FPS);
				int totalFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
				cv::Size const frame_size = cur_frame.size();
				cv::VideoWriter output_video;
				if (save_output_videofile)
					output_video.open(out_videofile, CV_FOURCC('D', 'I', 'V', 'X'), std::max(30, video_fps), frame_size, true);
				while (!cur_frame.empty())
				{
					auto frame_no = 0;
					m_nCurrentFrameNum = m_nFrameCount++;
					if (m_nCurrentFrameNum == 0)
						//주간,야간 표시
						day_or_night = decide_day_and_night(cur_frame);
					//if (m_nCurrentFrameNum == totalFrame - 1)
					//{
					//	//valueDestroy();
					//	std::cout << "Video Ended....." << std::endl;
					//	//output_video.release();
					//	//cap.release();
					//	//exit(0);
					//	//return 1;
					//}
					// always sync
					if (t_cap.joinable()) {
						t_cap.join();
						++fps_cap_counter;
						cur_frame = cap_frame.clone();
					}
					t_cap = std::thread([&]() { cap >> cap_frame; });
					++cur_time_extrapolate;

					// swap result bouned-boxes and input-frame
					if (consumed)
					{
						std::unique_lock<std::mutex> lock(mtx);
						det_image = detector.mat_to_image_resize(cur_frame);
						auto old_result_vec = detector.tracking_id(result_vec);
						auto detected_result_vec = thread_result_vec;
						result_vec = detected_result_vec;

#ifdef TRACK_OPTFLOW
						tracker_flow.update_cur_bbox_vec(result_vec);
						result_vec = tracker_flow.tracking_flow(cur_frame, true);    // track optical flow
#endif
						consumed = false;
						cv_pre_tracked.notify_all();
					}
					// launch thread once - Detection
					if (!t_detect.joinable()) {
						t_detect = std::thread([&]() {
							auto current_image = det_image;
							consumed = true;
							while (current_image.use_count() > 0 && !exit_flag) {
								auto result = detector.detect_resized(*current_image, frame_size.width,
									frame_size.height, thresh, false);    // true
								++fps_det_counter;
								std::unique_lock<std::mutex> lock(mtx);
								thread_result_vec = result;
								consumed = true;
								cv_detected.notify_all();
								if (detector.wait_stream) {
									while (consumed && !exit_flag) cv_pre_tracked.wait(lock);
								}
								current_image = det_image;
							}
						});
					}
					//while (!consumed);    // sync detection

					if (!cur_frame.empty()) {
						steady_end = std::chrono::steady_clock::now();
						if (std::chrono::duration<double>(steady_end - steady_start).count() >= 1) {
							current_det_fps = fps_det_counter;
							current_cap_fps = fps_cap_counter;
							steady_start = steady_end;
							fps_det_counter = 0;
							fps_cap_counter = 0;
						}

						large_preview.set(cur_frame, result_vec);
#ifdef TRACK_OPTFLOW
						++passed_flow_frames;
						track_optflow_queue.push(cur_frame.clone());
						result_vec = tracker_flow.tracking_flow(cur_frame);    // track optical flow
						extrapolate_coords.update_result(result_vec, cur_time_extrapolate);
						small_preview.draw(cur_frame, show_small_boxes);
#endif
						auto result_vec_draw = result_vec;
						if (extrapolate_flag) {
							result_vec_draw = extrapolate_coords.predict(cur_time_extrapolate);
							cv::putText(cur_frame, "extrapolate", cv::Point2f(10, 40),
								cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50, 50, 0), 2);
						}
						draw_boxes(cur_frame, result_vec_draw, obj_names, current_det_fps, current_cap_fps);
						//show_console_result(result_vec, obj_names);
						//large_preview.draw(cur_frame);

						auto key = cv::waitKey(3);    // 3 or 16ms
						if (key == 'f') show_small_boxes = !show_small_boxes;
						if (key == 'p') while (true) if (cv::waitKey(100) == 'p') break;
						if (key == 'e') extrapolate_flag = !extrapolate_flag;
						if (key == 27) { exit_flag = true; break; }

						if (output_video.isOpened() && videowrite_ready) {
							if (t_videowrite.joinable()) t_videowrite.join();
							write_frame = cur_frame.clone();
							videowrite_ready = false;
							t_videowrite = std::thread([&]() {
								output_video << write_frame; videowrite_ready = true;
							});
						}
					}

#ifndef TRACK_OPTFLOW
					// wait detection result for video-file only (not for net-cam)
					if (protocol != "rtsp://" && protocol != "http://" && protocol != "https:/") {
						std::unique_lock<std::mutex> lock(mtx);
						while (!consumed) cv_detected.wait(lock);
					}
#endif
				}
				exit_flag = true;
				if (t_cap.joinable()) t_cap.join();
				if (t_detect.joinable()) t_detect.join();
				if (t_videowrite.joinable()) t_videowrite.join();
				std::cout << "Video ended \n";
				break;
			}
			else if (file_ext == "txt") {    // list of image files
				FILE *resulf = fopen("../all_valid_result.txt", "a+");
				fprintf(resulf, "%s\t", filename.c_str());
				std::ifstream file(filename);
				std::string open_file_name = filename.substr(filename.find_last_of("\\") + 1);

				open_file_name = open_file_name.substr(0, open_file_name.find("."));
				open_file_name.append(".avi");
				int nResult = 0;
				if ((nResult = _access("SPC_result", 0)) == -1)
					_mkdir("SPC_result");
				if (!file.is_open()) std::cout << "File not found! \n";
				else
				{
					cv::VideoCapture open_video(open_file_name);
					char label_path[4096] = { '\0', };
					char img_path[4096] = { '\0' };
					auto frame_no = 1;
					auto current_TP = 0;
					auto current_FP = 0;
					std::string result_name = filename.erase(filename.find("."), filename.size());
					result_name = result_name + "_result.txt";
					FILE* fp = fopen(result_name.c_str(), "w");
					//FILE* fp = fopen("VALIDATION_RESULT.txt", "w");
					system("cls");
					std::cout << std::endl;
					fprintf(stderr, "%7s\t%5s\t%2s\t%12s\t%13s\n", "FrameNo",
						"GT", "ST", "TruepPositive", "FalsePositive");
					fprintf(fp, "%7s\t%5s\t%2s\t%12s\t%13s\n", "FrameNo",
						"GT", "ST", "TruepPositive", "FalsePositive");

					for (std::string line; file >> line;) {
						cv::Mat mat_img;
						open_video >> mat_img;
						day_or_night = decide_day_and_night(mat_img);
						std::fill_n(label_path, 4096, '\0');
						std::fill_n(img_path, 4096, '\0');
						char* path = new char[line.size() + 1];
						std::copy(line.begin(), line.end(), path);
						path[line.size()] = '\0';

						std::copy(line.begin(), line.end(), img_path);
						//find_replace(path, "image", "labels", imgpath);
						find_replace(img_path, ".jpg", ".txt", label_path);
						find_replace(label_path, ".JPEG", ".txt", label_path);
						//cv::Mat mat_img = cv::imread(line);
						Display_StaticReferenceMask_Validation(mat_img);
						set_StaticReferenceMask();
						cv::Size const frame_size = mat_img.size();
						auto det_image = detector.mat_to_image_resize(mat_img);
						auto result_vec = detector.detect_resized(*det_image, frame_size.width, frame_size.height,
							thresh, false);
						//std::vector<bbox_t> result_vec = detector.detect(mat_img);
						////GT로드
						if (result_vec.size() == 0)
						{
							Display_StaticReferenceMask_Validation(mat_img);

							fprintf(stderr, "%7d\t%5d\t%2d\t%12d\t%13d\n", frame_no, 0, 0, 0, 0);
							fprintf(fp, "%7d\t%5d\t%2d\t%12d\t%13d\n", frame_no, 0, 0, 0, 0);
							current_TP += 0;
							current_FP += 0;
						}
						else
						{
							int num_labels = 0;
							truth_box* truth = read_truth_boxes(label_path, &num_labels);
							convert_gt_coordinate(mat_img, truth, num_labels);

							draw_boxes_validation(mat_img, truth, result_vec, num_labels, fp, frame_no,
								&current_TP, &current_FP);
						}

						//delete[] truth;
						frame_no++;
						//draw_box_test(mat_img, result_vec, obj_names);
						/*	show_console_result(result_vec, obj_names);
						draw_boxes_validation(mat_img, result_vec, obj_names);*/
						//draw_boxes(mat_img, result_vec, obj_names);
						line.erase(0, 17);
						cv::imshow("Result", mat_img);
						cv::waitKey(3);
#ifdef SAVE_IMAGE
						std::vector<int> qualityType;
						qualityType.push_back(CV_IMWRITE_JPEG_QUALITY);
						qualityType.push_back(50);
						cv::imwrite("SPC_result/" + line, mat_img,qualityType);
#endif
					}
					fprintf(stderr, "%s", "=========================================================\n");
					fprintf(fp, "%s", "=========================================================\n");
					fprintf(stderr, "%7s\t%5s\t%2s\t%12d\t%13d\n", "Total",
						"", "", current_TP, current_FP);
					fprintf(fp, "%7s\t%5s\t%2s\t%12d\t%13d\n", "Total",
						"", "", current_TP, current_FP);
					fprintf(stderr, "%7s\t%5s\t%2s\t%12s\t%13.2f\n", "Precision",
						"", "", "", (current_TP*1.0 / (current_TP + current_FP)*100.0));
					fprintf(fp, "%7s\t%5s\t%2s\t%12s\t%13.2f\n", "Precision",
						"", "", "", (current_TP*1.0 / (current_TP + current_FP)*100.0));
					fprintf(fp, "%15s : %3.lf\t%15s : %3.lf\t%15s : %.6lf\n", "Normal GT", normal_gt_count,
						"TP Normal ST", normal_st_count, "Ratio", normal_st_count / normal_gt_count);
					fprintf(fp, "%15s : %3.lf\t%15s : %3.lf\t%15s : %.6lf\n", "Caution GT", caution_gt_count,
						"TP Caution ST", caution_st_count, "Ratio", caution_st_count / caution_gt_count);
					fprintf(fp, "%15s : %3.lf\t%15s : %3.lf\t%15s : %.6lf\n", "Warning GT", warning_gt_count,
						"TP Warning ST", warning_st_count, "Ratio", warning_st_count / warning_gt_count);
					std::cout << "Validtaion Completed" << std::endl;
					valueDestroy();
					fprintf(resulf, "%3.lf\n", (current_TP*1.0 / (current_TP + current_FP)*100.0));
					fclose(fp);
					fclose(resulf);
					exit(0);
				}
			}
			else {    // image file
				cv::Mat mat_img = cv::imread(filename);

				auto start = std::chrono::steady_clock::now();
				std::vector<bbox_t> result_vec = detector.detect(mat_img);
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> spent = end - start;
				std::cout << " Time: " << spent.count() << " sec \n";

				//result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required
				draw_boxes(mat_img, result_vec, obj_names);
				cv::imshow("window name", mat_img);
				show_console_result(result_vec, obj_names);
				cv::waitKey(0);
			}
#else
			//std::vector<bbox_t> result_vec = detector.detect(filename);

			auto img = detector.load_image(filename);
			std::vector<bbox_t> result_vec = detector.detect(img);
			detector.free_image(img);
			show_console_result(result_vec, obj_names);
#endif
		}
		catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
		catch (...) { std::cerr << "unknown exception \n"; getchar(); }
		filename.clear();
	}
}

void DetectionProcessing::Display_referenceMask(cv::Mat current_Img)
{
	//optFlow.get_ReferenceLine_Mask(m_mask);
	//optFlow.set_ReferenceLine_Mask(0, 0);
	// Optical Flow 특징 추출 프레임 여부 (우측 하단 사각박스)
	/*if(m_nCurrnetFrameNum == m_nPrevFrameNum)
	{
	int h = current_Img.cols;
	int w = current_Img.rows;

	int x, y = 0;
	x = 10;
	y = 10;
	current_Img.at<cv::Vec3b>(h-y, w-x)[0] = current_Img.at<cv::Vec3b>(h-y,w-x+1)[0] = current_Img.at<cv::Vec3b>(h-y, w-x+2)[0] =
	current_Img.at<cv::Vec3b>(h-y, w-x+3)[0] = 0;
	current_Img.at<cv::Vec3b>(h-y, w-x)[1] = current_Img.at<cv::Vec3b>(h-y, w-x+1)[1] = current_Img.at<cv::Vec3b>(h-y, w-x+2)[1] =
	current_Img.at<cv::Vec3b>(h-y, w-x+3)[1] = 0;
	current_Img.at<cv::Vec3b>(h-y, w-x)[2] = current_Img.at<cv::Vec3b>(h-y, w-x+1)[2] = current_Img.at<cv::Vec3b>(h-y, w-x+2)[2] =
	current_Img.at<cv::Vec3b>(h-y, w-x+3)[2] = 255;

	y = 11;
	current_Img.at<cv::Vec3b>(h - y, w - x)[0] = current_Img.at<cv::Vec3b>(h - y, w - x + 1)[0] = current_Img.at<cv::Vec3b>(h - y, w - x + 2)[0] =
	current_Img.at<cv::Vec3b>(h - y, w - x + 3)[0] = 0;
	current_Img.at<cv::Vec3b>(h - y, w - x)[1] = current_Img.at<cv::Vec3b>(h - y, w - x + 1)[1] = current_Img.at<cv::Vec3b>(h - y, w - x + 2)[1] =
	current_Img.at<cv::Vec3b>(h - y, w - x + 3)[1] = 0;
	current_Img.at<cv::Vec3b>(h - y, w - x)[2] = current_Img.at<cv::Vec3b>(h - y, w - x + 1)[2] = current_Img.at<cv::Vec3b>(h - y, w - x + 2)[2] =
	current_Img.at<cv::Vec3b>(h - y, w - x + 3)[2] = 255;

	y = 12;
	current_Img.at<cv::Vec3b>(h - y, w - x)[0] = current_Img.at<cv::Vec3b>(h - y, w - x + 1)[0] = current_Img.at<cv::Vec3b>(h - y, w - x + 2)[0] =
	current_Img.at<cv::Vec3b>(h - y, w - x + 3)[0] = 0;
	current_Img.at<cv::Vec3b>(h - y, w - x)[1] = current_Img.at<cv::Vec3b>(h - y, w - x + 1)[1] = current_Img.at<cv::Vec3b>(h - y, w - x + 2)[1] =
	current_Img.at<cv::Vec3b>(h - y, w - x + 3)[1] = 0;
	current_Img.at<cv::Vec3b>(h - y, w - x)[2] = current_Img.at<cv::Vec3b>(h - y, w - x + 1)[2] = current_Img.at<cv::Vec3b>(h - y, w - x + 2)[2] =
	current_Img.at<cv::Vec3b>(h - y, w - x + 3)[2] = 255;
	}*/

	for (int h = 0; h < current_Img.rows - 1; h++)
	{
		for (int w = 0; w < current_Img.cols - 1; w++)
		{
			if (abs(m_mask[h][w] - m_mask[h][w + 1]) != 0 || abs(m_mask[h][w] - m_mask[h + 1][w]) != 0)
			{
				current_Img.at<cv::Vec3b>(h, w)[0] = 0;
				current_Img.at<cv::Vec3b>(h, w)[1] = 255;
				current_Img.at<cv::Vec3b>(h, w)[2] = 0;
			}
		}
	}
}

/**
@brief 검출된 보행자 윈도우의 위험 요소 특징  추출
*/

void DetectionProcessing::ExtractSuddenCrossingFeatureInCropedObject()
{
}

/*
*	@brief		갑자기 건너는 보행자 검출
@param		result_vec			: YOLO 에서 검출된 보행자 정보
*	@param		nDetectionCount		: 최종 검출 보행자 수
*	@param		optFeatures			: optical Flow 결과 포인트
*	@param		nOptCount			: optical Flow 결과 포인트 개수
*	@param		nVehicleDirection	: 자동차 진행방향
*/
void DetectionProcessing::detectionSuddenCrossingPedestrian(cv::Mat currImg, std::vector<bbox_t> result_vec)

{
	//갑자기 건너는 보행자 검출 2가지
	//1) Reference Mask영역에 100% 포함되면 무조건 검출
	//2) Refernce Mask영역의 Line부분에 걸치면 다음 조건 검사
	//	겹침 정도 & 이동방향 & 이동크기
	//	이동방향 : Line에 걸쳐져 있는 윈도우들은 영상을 이등분해서 왼쪽 파트 영상에서는 ->, 오른쪽 파트 영상에서는 <- 방향으로 이동하는 보행자를 검출

	int k, f;

	double dDecision_OverlappedRate = 0.0;
	double dDecision_MovementDir = 0.0;
	double dDecision_MovementMag = 0.0;
	double dDecision_Distance = 0.0;
	double dDecision_Overlapped_prob = 0.0;
	double dDecision_MovementDir_prob = 0.0;
	double dDecision_MovementMag_prob = 0.0;
	double dDecision_Distance_prob = 0.0;
	double dExceptionalRiskValue = 0.0;			// 예외적 위험상황에 대한 가중치 (2015-01-19)
	int nDistanceFromCam = 100;					// 보행자 윈도우 ~ 카메라간 거리 (2015-01-19)
	int nWindowPositionIn3part = 0;				// 보행자 윈도우의 Reference Line 기준으로 위치 (0:중심, 1:Left RL 겹침 위치, 2:Right RL 겹침 위치)
	double dResult_prob = 0;

	int nTmpCount = 0;
	int x, y;
	int detPeopleCount = 0;
	//검출된 보행자 수
	for (auto &i : result_vec)
		//detection 결과가 person일때만
		if (true)
			//if (i.obj_id == 0)
			detPeopleCount++;

#ifdef DISPLAY_PED_COUNT
	// 검출된 보행자 수 출력
	char output_count[3] = { 0, };
	snprintf(output_count, 3, "%d", detPeopleCount);
	putText(currImg, output_count, cv::Point2f(IMG_SIZE_WIDTH*0.078, IMG_SIZE_HEIGHT*0.166120),
		cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(0, 0, 255), 2);
#endif
	//YOLO에 의해 검출된 보행자 윈도우 정보를 담을 구조체

	DetPeople_INFO *detectedPeopleInfo;
	if (detPeopleCount > 0)
		detectedPeopleInfo = new DetPeople_INFO[detPeopleCount];
	else
	{
		detectedPeopleInfo = new DetPeople_INFO[1];
		detectedPeopleInfo[0] = {};
	}
	int m = 0;
	for (auto &i : result_vec)
	{
		if (true)
			//if (i.obj_id == 0)
		{
			// 보행자 횡방향 : 좌, 우, 센터
			int rowDirection = IMG_SIZE_WIDTH / 3;
			int centerX = i.x + i.w / 2;
			cv::Point2f rowDirection_position;
			rowDirection_position.x = i.x + i.w - 30;
			rowDirection_position.y = i.y + i.h + 55;
			int pedestrian_Position = 0;

			detectedPeopleInfo[m].det_id = m;
			///////////////영상내에서 보행자가 위치한 영역/////////////////////////////////

			if (0 <= centerX && centerX < IMG_SIZE_WIDTH * 0.4)								// Left
			{
				detectedPeopleInfo[m]._prt_position = 0x02;
				pedestrian_Position = 0;
			}
			else if (IMG_SIZE_WIDTH * 0.4 <= centerX && centerX < IMG_SIZE_WIDTH * 0.6)		// Center
			{
				detectedPeopleInfo[m]._prt_position = 0x01;
				pedestrian_Position = 1;
			}
			else if (IMG_SIZE_WIDTH*0.6 <= centerX && centerX <= IMG_SIZE_WIDTH)				// Right
			{
				detectedPeopleInfo[m]._prt_position = 0x03;
				pedestrian_Position = 2;
			}
#ifdef DISPLAY_POSITION
			if (pedestrian_Position == 0)								// Left
				putText(currImg, "Left", rowDirection_position, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 0, 0), 2);
			else if (pedestrian_Position == 1)
				putText(currImg, "Center", rowDirection_position, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 0, 0), 2);
			else if (pedestrian_Position = 2)
				putText(currImg, "Right", rowDirection_position, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 0, 0), 2);
			putText(currImg, std::to_string(detectedPeopleInfo[m].det_id), cv::Point(rowDirection_position.x - 30, rowDirection_position.y),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 120), 2);
#endif
			detectedPeopleInfo[m].start_x = i.x;
			detectedPeopleInfo[m].start_y = i.y;
			detectedPeopleInfo[m].end_x = i.x + i.w;
			detectedPeopleInfo[m].end_y = i.y + i.h;
			detectedPeopleInfo[m].obj_class = i.obj_id;

			//2017-12-13 차량으로부터의 사람 거리 출력
			CvPoint distance_point;
			distance_point.x = i.x;
			distance_point.y = i.y - 40;
			double people_height = i.h;
			double people_distance = REAL_PEOPLE_HEIGHT * FOCAL_LENGTH / people_height;
			char output_distance[8];
			snprintf(output_distance, 8, "%.2lf m", people_distance);
#ifdef DISPLAY_DISTANCE
			putText(currImg, output_distance, distance_point, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(14, 201, 255), 2);
#endif
			detectedPeopleInfo[m].distance = people_distance;

			int cvtDistance = detectedPeopleInfo[m].distance / 5;
			if (cvtDistance == 0)		// 거리가 5m미만인경우는 전부 5m로 가정
				detectedPeopleInfo[m]._prt_distance = 0x01;
			else if (cvtDistance > 8)	// 거리가 40m이상인경우
				detectedPeopleInfo[m]._prt_distance = 0x09;
			else
				detectedPeopleInfo[m]._prt_distance = cvtDistance;
			m++;
		}
	}

	//YOLO에 의해 검출된 보행자 윈도우중 1개의 윈도우 정보
	DetPeople_INFO current_winInfo;

	//영상 전체의 Optical Flow의 평균 이동 속도

	//m_dAvgOptSpeedofImage = m_dAvgOptSpeedofImage / nTmpCount;
	for (k = 0; k < detPeopleCount; k++)
	{
		detectedPeopleInfo[k].nSuddenCrossing = 0;

		current_winInfo.start_x = detectedPeopleInfo[k].start_x;
		current_winInfo.start_y = detectedPeopleInfo[k].start_y;
		current_winInfo.end_x = detectedPeopleInfo[k].end_x;
		current_winInfo.end_y = detectedPeopleInfo[k].end_y;
		current_winInfo.det_id = detectedPeopleInfo[k].det_id;
		int ped_orient = detectedPeopleInfo[k].obj_class;
		int ped_position;
		int ped_coord_x = current_winInfo.start_x + (current_winInfo.end_x - current_winInfo.start_x) / 2;

		if (currImg.cols*0.3671 <= ped_coord_x && ped_coord_x <= currImg.cols*0.5156)
			ped_position = 0;
		else if (0 <= ped_coord_x && ped_coord_x < currImg.cols*0.3671)
			ped_position = 1;
		else
			ped_position = 2;
		double ped_distance = detectedPeopleInfo[k].distance;
		//보행자 방향 표시
#ifdef DISPLAY_ORIENTATION
		orientation_display(currImg, ped_orient, detectedPeopleInfo[k]);
#endif
		dDecision_OverlappedRate = 0.0;
		// 조건 (1) 체크 : 레퍼런스 라인과 보행자 윈도우간의 겹칩율
		dDecision_OverlappedRate = detectionSuddenCrossing_overlappedAreaRate(current_winInfo.start_x, current_winInfo.end_x,
			current_winInfo.start_y, current_winInfo.end_y, nWindowPositionIn3part);

		// 조건 (2), (3) 체크
		//nDistanceFromCam = extractDistanceBetweenWinNCam(current_winInfo.end_y - current_winInfo.start_y);

		detectionSuddenCrossing_movement(currImg, ped_orient, ped_position, ped_distance,
			dDecision_MovementDir, dDecision_MovementMag);

		//moveDirection = detectionSuddenCrossing_movement(currImg, &current_winInfo,
		//	//optFeatures,
		//	nOptCount, m_dAvgOptSpeedofImage, nVehicleDirection,
		//	nWindowPositionIn3part, nDistanceFromCam, dDecision_MovementDir, dDecision_MovementMag);

		if (ped_orient == 3 || ped_orient == 4 || ped_orient == 5)			//  보행자의 움직임방향 :  오른쪽->왼쪽
		{
			detectedPeopleInfo[k]._prt_direction = 0x03;
			/*			cv::arrowedLine(currImg, cv::Point(current_winInfo.end_x, current_winInfo.start_y - 10),
							cv::Point(current_winInfo.start_x, current_winInfo.start_y - 10), cv::Scalar(255, 255, 255), 2, 8, 0, 0.3)*/;
		}
		else if (ped_orient == 1 || ped_orient == 0 || ped_orient == 7)		// 보행자의 움직임방향 : 왼쪽 -> 오른쪽
		{
			detectedPeopleInfo[k]._prt_direction = 0x02;
			/*			cv::arrowedLine(currImg, cv::Point(current_winInfo.start_x, current_winInfo.start_y - 10),
							cv::Point(current_winInfo.end_x, current_winInfo.start_y - 10), cv::Scalar(255, 255, 255), 2, 8, 0, 0.3)*/;
		}
		//판단 보류
		else
			detectedPeopleInfo[k]._prt_direction = 0x04;
		// 2015-01-19 보행자 윈도우가 자동차와 매우 근접해 있고(윈도우높이 130이상), 겹침비율이 0.4이상,
		//motion score 0.2 이상 => 예외적 위험 상황 발생으로 가중치 부여
		/*if (dDecision_OverlappedRate > 0.3 && dDecision_MovementDir > 0.2 && abs(current_winInfo.end_y - current_winInfo.start_y) > 130)
		dExceptionalRiskValue = 0.35;*/

		//거리가 가까울수록 높은 가중치 값
		dDecision_Distance_prob = 0.0;
		dDecision_Overlapped_prob = 0.0;
		dDecision_MovementDir_prob = 0.0;
		dDecision_MovementMag_prob = 0.0;

		dDecision_Distance_prob = 1.0 - detectedPeopleInfo[k].distance / MAX_DISTANCE;
		//dDecision_Distance_prob = detectedPeopleInfo[k].distance / MAX_DISTANCE;
		// 겹침비율, 움직임방향, 움직임크기와 가우시안 분포를 사용한 최종 보행자 검출
		dDecision_Distance_prob =
			detectionGaussianProbabilityDensityFunction(1 - dDecision_Distance_prob, 0.0, 0.5);
		//test//
		//dDecision_OverlappedRate = (dDecision_OverlappedRate *detectedPeopleInfo[k].distance/ MAX_DISTANCE);
		dDecision_Overlapped_prob =
			detectionGaussianProbabilityDensityFunction(1.0 - dDecision_OverlappedRate, 0.0, 0.6);
		//detectionGaussianProbabilityDensityFunction(dDecision_OverlappedRate, 0.0, 0.6);
		dDecision_MovementDir_prob =
			detectionGaussianProbabilityDensityFunction(1.0 - dDecision_MovementDir, 0.0, 0.5);
		dDecision_MovementMag_prob =
			detectionGaussianProbabilityDensityFunction(1.0 - dDecision_MovementMag, 0.0, 0.5);
		//dResult_prob = dDecision_Distance_prob*0.15 + dDecision_Overlapped_prob*0.6 + dDecision_MovementDir_prob*0.2
		//	+ dDecision_MovementMag_prob*0.1 + dExceptionalRiskValue;		// 2015-01-19	예외적 위험 상황에 대한 가중치 부여
		//dResult_prob = dDecision_Distance_prob*0.25 + dDecision_Overlapped_prob*0.65 + dDecision_MovementDir_prob*0.1;


		//dResult_prob = dDecision_Distance_prob*0.3 + dDecision_Overlapped_prob*0.6 + dDecision_MovementDir_prob*0.1;
		if (detectedPeopleInfo[k].distance <= 5.0)
			dResult_prob = (dDecision_Distance_prob - 0.4)*0.05 + dDecision_Overlapped_prob*0.9 + dDecision_MovementDir_prob*0.05;
		else if (5.0 <= detectedPeopleInfo[k].distance && detectedPeopleInfo[k].distance <= 10.0)
			dResult_prob = (dDecision_Distance_prob - 0.4)*0.6 + dDecision_Overlapped_prob*0.3 + dDecision_MovementDir_prob*0.1;
		else
			dResult_prob = (dDecision_Distance_prob - 0.4)*0.7 + dDecision_Overlapped_prob*0.2 + dDecision_MovementDir_prob*0.1;

		//예외적으로 차량주행방향에 보행자가 위치할 때 가중치 추가
		double center_x = detectedPeopleInfo[k].start_x + ((detectedPeopleInfo[k].end_x - detectedPeopleInfo[k].start_x) / 2);
		if (detectedPeopleInfo[k].distance <= 10.0 &&
			IMG_SIZE_WIDTH*0.46875 <= center_x && center_x <= IMG_SIZE_WIDTH*0.65625)
				if (dResult_prob + 0.2 <= 1.0)
					dResult_prob += 0.2;
		//차량 주행 반대 차선에서 주행차선으로 들어오는 보행자
		else if (detectedPeopleInfo[k].distance <= 10.0 &&
			IMG_SIZE_WIDTH*0.3359 <= center_x && center_x < IMG_SIZE_WIDTH*0.46875)
				dResult_prob += 0.1;
		else
			if (dResult_prob - 0.1 > 0)
				dResult_prob -= 0.1;
		
		//
		/*dResult_prob = dDecision_Distance_prob*0.3 + dDecision_Overlapped_prob*0.4 + dDecision_MovementDir_prob*0.3;*/
		/*std::cout.precision(6);
		std::cout << std::fixed << current_winInfo.det_id << " : " << dDecision_Distance_prob << "\t" <<
			dDecision_Overlapped_prob << "\t" << dDecision_MovementDir_prob << "\t" << dDecision_MovementMag_prob << "\t" << dResult_prob << std::endl;*/
			//ReferenceLine에 전혀 겹치지 않는경우 항상 Normal

			//if (dDecision_OverlappedRate == 0)
		if (dDecision_Overlapped_prob < 0.36)
		{
			int box_x, box_y, box_w, box_h;

			box_x = detectedPeopleInfo[k].start_x;
			box_y = detectedPeopleInfo[k].start_y;
			box_w = detectedPeopleInfo[k].end_x - detectedPeopleInfo[k].start_x;
			box_h = detectedPeopleInfo[k].end_y - detectedPeopleInfo[k].start_y;

#ifdef DISPLAY_PROBABILITY
			putText(currImg, std::to_string(dResult_prob), cv::Point(box_x + box_w - 30, box_y - 30),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);
			detectedPeopleInfo[k]._prt_riskRate = detectedPeopleInfo[k].dProbWarning * 100;

			cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(0, 255, 0), 1);
			putText(currImg, "Normal", cv::Point(box_x + box_w - 30, box_y + box_h + 30),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);
#endif
			detectedPeopleInfo[k]._prt_risk = 0x01;
		}
		else
		{
			double dis = dDecision_Distance_prob *0.25;
			double overl = dDecision_Overlapped_prob *0.5;
			double MovDir = dDecision_MovementDir_prob * 0.15;
			double MovMag = dDecision_MovementMag_prob * 0.1;

			if (dResult_prob > 1.0)
				dResult_prob = 1.0;
			if (dResult_prob > 0.65)
			{
				//detectedPeopleInfo[k].dProbWarning = 1.0;		// 보행자 위험정도 저장(화면 표시)
				detectedPeopleInfo[k].nSuddenCrossing = 1;
				detectedPeopleInfo[k].dProbWarning = dResult_prob;
			}
			else
				detectedPeopleInfo[k].dProbWarning = dResult_prob;

			//Normal, Caution, Warning 상태에 따른 검출 윈도우 출력
			int box_x, box_y, box_w, box_h;

			box_x = detectedPeopleInfo[k].start_x;
			box_y = detectedPeopleInfo[k].start_y;
			box_w = detectedPeopleInfo[k].end_x - detectedPeopleInfo[k].start_x;
			box_h = detectedPeopleInfo[k].end_y - detectedPeopleInfo[k].start_y;

			////각 위험도 변수값
			//putText(currImg, std::to_string(dis), cv::Point(box_x + box_w + 30, box_y), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);
			//putText(currImg, std::to_string(overl), cv::Point(box_x + box_w + 30, box_y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);
			//putText(currImg, std::to_string(MovDir), cv::Point(box_x + box_w + 30, box_y + 28), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);
			//putText(currImg, std::to_string(MovMag), cv::Point(box_x + box_w + 30, box_y + 39), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);

			//위험도 스코어 표시
#ifdef DISPLAY_SPC_SCORE
			putText(currImg, std::to_string(dResult_prob), cv::Point(box_x + box_w - 30, box_y - 30),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);
#endif
			detectedPeopleInfo[k]._prt_riskRate = detectedPeopleInfo[k].dProbWarning * 100;

			////Normal
			//if (detectedPeopleInfo[k].dProbWarning < 0.53)
			//{
			//	cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(0, 255, 0), 1);
			//	putText(currImg, "Normal", cv::Point(box_x + box_w - 30, box_y + box_h + 30),
			//		cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);
			//	detectedPeopleInfo[k]._prt_risk = 0x01;
			//}
			////Caution
			//else if (0.53 < detectedPeopleInfo[k].dProbWarning &&detectedPeopleInfo[k].dProbWarning < 0.695)
			//{
			//	cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(40, 127, 255), 2);
			//	putText(currImg, "Caution", cv::Point(box_x + box_w - 30, box_y + box_h + 30),
			//		cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(40, 127, 255), 2);
			//	detectedPeopleInfo[k]._prt_risk = 0x02;
			//}
			////Warning
			//else
			//{
			//	cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(0, 0, 255), 3);
			//	putText(currImg, "Warning", cv::Point(box_x + box_w - 30, box_y + box_h + 30),
			//		cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 3);
			//	detectedPeopleInfo[k]._prt_risk = 0x03;
			//}
			//Normal
			if (dResult_prob < 0.41)
			{
#ifdef DISPLAY_DANGEROUS_BOX
				cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(0, 255, 0), 1);
#endif
#ifdef DISPLAY_DANGEROUS
				putText(currImg, "Normal", cv::Point(box_x + box_w - 30, box_y + box_h + 30),
					cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);
#endif
				detectedPeopleInfo[k]._prt_risk = 0x01;
			}
			//Caution
			//else if (0.33 < dDecision_OverlappedRate && dDecision_OverlappedRate < 0.695)
			else if (0.41 < dResult_prob && dResult_prob < 0.695)
			{
#ifdef DISPLAY_DANGEROUS_BOX
				cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(40, 127, 255), 2);
#endif
#ifdef DISPLAY_DANGEROUS
				putText(currImg, "Caution", cv::Point(box_x + box_w - 30, box_y + box_h + 30),
					cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(40, 127, 255), 2);
#endif
				detectedPeopleInfo[k]._prt_risk = 0x02;
			}
			//Warning
			else
			{
#ifdef DISPLAY_DANGEROUS_BOX
				cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(0, 0, 255), 3);
#endif
#ifdef DISPLAY_DANGEROUS
				putText(currImg, "Warning", cv::Point(box_x + box_w - 30, box_y + box_h + 30),
					cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 3);
#endif
				detectedPeopleInfo[k]._prt_risk = 0x03;
			}
		}
	}

	/*DetPeople_INFO *detectedPeopleInfo = new DetPeople_INFO[detPeopleCount];
	memset(&detectedPeopleInfo, 0, sizeof(DetPeople_INFO));
	for (int a = 0; a < detPeopleCount; a++)
		memcpy(&detectedPeopleInfo[a], &detectedPeopleInfo[a], sizeof(DetPeople_INFO));*/
	

	//결과 이미지 저장
	/*char fileName[5];
	char filePath[256] = "ResultImage/";
	char saveImage[256];
	sprintf(saveImage, "%s%d.jpg", filePath, m_nCurrnetFrameNum);
	cv::imwrite(saveImage, currImg);*/
	convertDataToProtocolMsg(detectedPeopleInfo, detPeopleCount, day_or_night);		//0 :주간, 1: 야간
	delete[] detectedPeopleInfo;
}

/*
*	@brief		갑자기 뛰어드는 보행자 검출 조건1 - Reference Line검사
*	@param		sy, ey : 윈도우 시작, 끝 y좌표
*	@param		sx, ex : 윈도우 시작, 끝 x좌표
*	@param		nWindowPosition : 현재 보행자 윈도우의 Reference Line기준의 위치(0 : 중심, 1 : Left, RL 겹침위치, 2 : Right RL 겹침 위치)
*	@return		Referecne Mask와 현재 윈도우와의 겹침 비율
*/
double DetectionProcessing::detectionSuddenCrossing_overlappedAreaRate(int start_x, int end_x, int start_y,
	int end_y, int& nWindowPosition)
{
	int i, j;
	double dOverlappedRate = 0.0;			// 현재 윈도우와 Reference Mask와 겹침 비율
	int nOverlappedArea = 0;				// 현재 윈도우와 Reference Mask와 겹침 비율
	int nLeftRL_pointCount = 0;				// 현재 윈도우와 Left Reference Line 겹침 픽셀 개수
	int nRightRL_pointCount = 0;			// 현재 윈도우와 Right Refrence Line 겹침 픽셀 개수

	if (end_x > IMG_SIZE_WIDTH)
		end_x = IMG_SIZE_WIDTH;
	if (end_y > IMG_SIZE_HEIGHT)
		end_y = IMG_SIZE_HEIGHT;
	int nWindowArea = (end_y - start_y) * (end_x - start_x);

	for (i = start_y; i < end_y; i++)
	{
		for (j = start_x; j < end_x; j++)
		{
			if (m_mask[i][j] != 0)
			{
				nOverlappedArea++;

				if (m_mask[i][j] == 100)
					nLeftRL_pointCount++;
				if (m_mask[i][j] == 200)
					nRightRL_pointCount++;
			}
		}
	}
	// Reference Mask 겹침 비율 계산
	dOverlappedRate = (double)nOverlappedArea / (double)nWindowArea;
	//Reference Mask와 겹침 비율이 0.6이하이고, Reference Line와 해당 윈도우가 겹쳐졌을 경우, 보행자 윈도우 위치 결정

	if (dOverlappedRate < 0.6 && (nLeftRL_pointCount != 0 || nRightRL_pointCount != 0))
	{
		if (nLeftRL_pointCount != 0 && (nLeftRL_pointCount > nRightRL_pointCount))
			nWindowPosition = 1;
		else if (nRightRL_pointCount != 0 && (nLeftRL_pointCount < nRightRL_pointCount))
			nWindowPosition = 2;
		else
			nWindowPosition = 0;
	}
	return dOverlappedRate;
}

/*
*	@brief		갑자기 뛰어드는 보행자 검출 조건2,3 - 이동 방향과 크기를 이용한 판단
*	@param		winInfo : 보행자 윈도우 정보
*				optFeatures : optical Flow 정보
*				nOptCount : optical Flow feature 포인트 개수
*				dOptSpeed : 해당 보행자 윈도우의 보행자 평균 속도
*				nVehicleDirection : 자동차 주행 방향
*				nWindowPositionIn3part : 보행자 윈도우 위치(0:중심, 1:Left Reference Line과 겹친 위치, 2: Right Reference Line과 겹친 위치)
*				dResultDirRate : 보행자 motion score
dResultMovingSpeedRate : 보행자 motion speed score
nDistanceFromCam : 보행자 윈도우와 자동차간 거리
*/
void DetectionProcessing::detectionSuddenCrossing_movement(cv::Mat currImg, int nDirIndex,
	int nWindowPositionIn3part, double nDistanceFromCam, double& dResultDir, double& dResultDistance)
{
	//double aDirScore[3][8] = { 1.0, 0.4, 0.1, 0.4, 1.0, 0.7, 0.1, 0.7,			// 윈도우가 RF 중심영역에 위치
	//	0.7, 0.1, 0.1, 0.4, 1.0, 0.7, 0.1, 0.1,			// 윈도우가 Left Reference Line과 겹쳐질 경우의 Direction Score
	//	1.0, 0.4, 0.1, 0.1, 0.7, 0.1, 0.1, 0.7 };		// 윈도우가 Right Reference Line과 겹쳐질 경우의 Direction Score
	double aDirScore[3][8] = { 0.4, 0.6, 0.9, 0.6, 0.4, 0.5, 0.7, 0.5,			// 윈도우가 RF 중심영역에 위치
		0.9, 0.7, 0.4, 0.2, 0.1, 0.2, 0.4, 0.7,			// 윈도우가 Left Reference Line과 겹쳐질 경우의 Direction Score
		0.1, 0.2, 0.4, 0.7, 0.9, 0.7, 0.4, 0.2 };		// 윈도우가 Right Reference Line과 겹쳐질 경우의 Direction Score

	int distance_factor[7] = { 1,2,3,4,5,6,7 };

	int factor_idx = nDistanceFromCam / 5;
	if (factor_idx == 0)
		factor_idx = 1;

	dResultDir = 0.0;
	dResultDistance = 0.0;

	//dScore_Sum += aDirScore[nWindowPositionIn3part][nDirIndex];			// x좌표 가중치에 의한 Score값 계산
	dResultDir = aDirScore[nWindowPositionIn3part][nDirIndex];			// x좌표 가중치에 의한 Score값 계산

	dResultDistance = 1 / (exp((nDistanceFromCam / MAX_DISTANCE)*(1 - dResultDir)*distance_factor[factor_idx - 1]));

	//////////////////////////
	// extract the distance : window를 이용한 보행자 카메라와의 거리 측정
	//////////////////////////

	/*for (f = 0; f < nOptCount; f++)
	std::cout << "optFeatures[" << f << "].angle : " << optFeatures[f].angle << std::endl;*/

	// 해당 Window의 평균 이동 방향과 크기 추출
	//Trace(_T("Mag(avg:%.2lf): "), dOptSpeed);
	//	for (f = 0; f<nOptCount; f++)
	//	{
	//		if (optFeatures[f].magnitude == -1)		// invalid point
	//		{
	//			continue;
	//		}
	//
	//		//////////////////////////////////////
	//		// vehicle's ego-motion 보완
	//		//////////////////////////////////////
	//
	//		// 현재 Widdow의 포함되는 optical Flow의 벡터 추출하여 이동방향과 크기 계산
	//		/*if (optFeatures[f].x >= winInfo->start_x  && optFeatures[f].x <= winInfo->end_x
	//		&& optFeatures[f].y >= winInfo->start_y && optFeatures[f].y <= winInfo->end_y)*/
	//		// 보행자 윈도우의 2/3지점까지만 optical flow체크
	//		if (optFeatures[f].x >= winInfo->start_x  && optFeatures[f].x <= winInfo->end_x
	//			&& optFeatures[f].y >= winInfo->start_y && optFeatures[f].y <= winInfo->end_y*0.6)
	//		{
	//			dTmpMag = optFeatures[f].magnitude;
	//
	//			// 좌회전이나 우회전일 경우, 자동차 움직임에 의한 Optical Flow 방향 & 크기를 조절
	//			if (nVehicleDirection == 1) // 좌회전
	//			{
	//				if ((optFeatures[f].angle >= 0 && optFeatures[f].angle < 70) ||
	//					(optFeatures[f].angle >= 290 && optFeatures[f].angle < 360))
	//					dTmpMag += dOptSpeed;
	//				else
	//					dTmpMag -= dOptSpeed;
	//			}
	//			else if (nVehicleDirection == 2)	// 우회전
	//			{
	//				if (optFeatures[f].angle >= 110 && optFeatures[f].angle < 250)
	//					dTmpMag += dOptSpeed;
	//				else
	//					dTmpMag -= dOptSpeed;
	//			}
	//
	//			//Trace(_T("(%d->%.2lf) "), optFeatures[f].magnitude, dTmpMag);
	//
	//			// 좌회전이나 우회전일 경우, 자동차 우직임에 의한 Optical Flow 방향 및 크기 조절
	//			if (dTmpMag <= 0.0)
	//			{
	//				optFeatures[f].angle = optFeatures[f].angle + 180;
	//				dTmpMag = dTmpMag * (-1.0); //optFeatures[f].magnitude + dOptSpeed;
	//			}
	//
	//			if (optFeatures[f].angle > 360)
	//				optFeatures[f].angle -= 360;
	//
	//			// 2014-05-27 이동 크기 3이하 검사 무시 : 3->2
	//			//if( dTmpMag < 2 )		// 이동 방향이 3이하이면 방향 검사에서 무시
	//			//	continue;
	//
	//			////////////////////////////////////////
	//			// pedestrian's moving speed ratio
	//			////////////////////////////////////////
	//			dMovingSpeedRatio_Sum += (1.0 - (1.0 / exp(nDistanceFromCam*dTmpMag*0.025)));
	//
	//			nvalidptCount_forDir++;
	//
	//
	//
	//			////////////////////////////////////////
	//			// pedestrian's moving direction ratio
	//			////////////////////////////////////////
	//
	//			if (optFeatures[f].angle >= 360)			optFeatures[f].angle = 0;
	//
	//			// Movement Direction 조건 : 움직임 방향에 따라 방향 index를 결정하고 그 방향에 대한 Score값을 선택
	//			// 방향은 0~360에서 45도씩 8방향으로 분할
	//			if ((optFeatures[f].angle) <= 23 || optFeatures[f].angle >= 337)
	//				nDirIndex = 0;
	//			else
	//				nDirIndex = (int)((optFeatures[f].angle - 23) / 45.0) + 1;			// 각도 Index 선택
	//
	//#if RESULT_DISPLAY == 1
	//																					//Right -> Left, Left-> Right로 이동중인 보행자 표시
	//			int lineType = 8;
	//			int tickness = 1;
	//			double tipLength = 0.1;
	//			auto angleRad_LR = 0 * CV_PI / 180.0;
	//			auto angleRad_RL = 180 * CV_PI / 180.0;
	//			auto length = winInfo->end_x - winInfo->start_x;
	//			auto direction_LR = cv::Point(length * cos(angleRad_LR), length*sin(angleRad_LR));
	//			auto direction_RL = cv::Point(length * cos(angleRad_RL), length*sin(angleRad_RL));
	//			auto startP_LR = cv::Point(winInfo->start_x, winInfo->start_y);
	//			auto startP_RL = cv::Point(winInfo->end_x, winInfo->start_y);
	//
	//
	//#endif
	//			dScore_Sum += aDirScore[nWindowPositionIn3part][nDirIndex];			// x좌표 가중치에 의한 Score값 계산
	//
	//		}	// end of if
	//	}	// end of f
	//		//Trace(_T("\n"));
	//
	//	if (nvalidptCount_forDir != 0)
	//	{
	//		dResultMovingSpeedRate = dMovingSpeedRatio_Sum / nvalidptCount_forDir;
	//		dResultDirRate = dScore_Sum / nvalidptCount_forDir;
	//
	//		// 2015-01-19 예외적 위험 상황 (자동차와 보행자가 매우 근접(윈도우크기<81) 하고, 모션 스코어가 0.4일 경우 예외적 위험 상황 판단 => 가중치 부여)
	//		/*if( nWindowWidth > aDistanceByWinWidth[0][1]-15.0 && dResultDirRate >= 0.4 )
	//		{
	//		Trace(_T("============ Risk Situation (Motion Score:%3.lf) \n"),dResultDirRate);
	//		dExceptionalRiskValue = 0.3;
	//		}*/
	//	}
	//	return nDirIndex;
	//	//if (nDirIndex == 1 || nDirIndex == 0 || nDirIndex == 7)					//Right -> Left
	//	//{
	//	//	cv::arrowedLine(currImg, cv::Point(winInfo->end_x, winInfo->start_y - 10), cv::Point(winInfo->start_x, winInfo->start_y - 10),
	//	//		cv::Scalar(255, 255, 255), 2, 8, 0, 0.3);
	//	//	return;
	//	//}
	//	//else
	//	//{
	//	//	cv::arrowedLine(currImg, cv::Point(winInfo->start_x, winInfo->start_y - 10), cv::Point(winInfo->end_x, winInfo->start_y - 10),
	//	//		cv::Scalar(255, 255, 255), 2, 8, 0, 0.3);
	//	//	return;
	//	//}
}

/*
*	@brief		가우시안 확률 밀도 함수
*/
double DetectionProcessing::detectionGaussianProbabilityDensityFunction(double value, double mean,
	double standardDeviation)
{
	return  exp(-(((value - mean)*(value - mean)) / (2 * standardDeviation*standardDeviation)));
}

/*
*	@brief		보행자 윈도우와 카메라간 거리 추출
*/
int DetectionProcessing::extractDistanceBetweenWinNCam(int nWindowHeight)
{
	int nDistance = 0;		// 실제 보행자~카메라 거리
	int aDistanceByWinHeight[6][2] = { 5,145,		//distance, window_height
		10,96,
		15,75,
		20,59,
		25,48,
		30,43 };
	for (int i = 0; i < 6; i++)
	{
		if (i == 0 && nWindowHeight >= aDistanceByWinHeight[0][1])
		{
			nDistance = aDistanceByWinHeight[0][0];
			break;
		}
		else if (i == 5 && nWindowHeight < aDistanceByWinHeight[5][1])
			nDistance = aDistanceByWinHeight[5][0];
		else
		{
			if (nWindowHeight < aDistanceByWinHeight[i - 1][1] && nWindowHeight >= aDistanceByWinHeight[i][1])
			{
				nDistance = aDistanceByWinHeight[i][0];
				break;
			}
		}
	}
	return nDistance;
}

/*
*	@brief		Static ReferenceLine Display
*/
void DetectionProcessing::Display_StaticReferenceMask(cv::Mat currImg, int day_or_night)
{
	cv::Point ptss[4];
#ifdef KATECH
	ptss[0] = referenceLine_ptss[0] = cv::Point(currImg.cols*0.12109375, currImg.rows*0.7847222);
	ptss[1] = referenceLine_ptss[1] = cv::Point(currImg.cols*0.4609375, currImg.rows*0.54861111);
	ptss[2] = referenceLine_ptss[2] = cv::Point(currImg.cols*0.5546875, currImg.rows*0.54861111);
	ptss[3] = referenceLine_ptss[3] = cv::Point(currImg.cols*0.9921875, currImg.rows*0.7847222);
#else
	//Day time
	if (day_or_night == 0)
	{
		ptss[0] = referenceLine_ptss[0] = cv::Point(currImg.cols*0.0006, currImg.rows*0.8194);
		ptss[1] = referenceLine_ptss[1] = cv::Point(currImg.cols*0.3671, currImg.rows*0.4861);
		ptss[2] = referenceLine_ptss[2] = cv::Point(currImg.cols*0.5156, currImg.rows*0.4861);
		ptss[3] = referenceLine_ptss[3] = cv::Point(currImg.cols*0.9296, currImg.rows*0.9027);
	}
	//Night time
	else
	{
		ptss[0] = referenceLine_ptss[0] = cv::Point(currImg.cols*0.0006, currImg.rows*0.8666);
		ptss[1] = referenceLine_ptss[1] = cv::Point(currImg.cols*0.3612, currImg.rows*0.625);
		ptss[2] = referenceLine_ptss[2] = cv::Point(currImg.cols*0.5159, currImg.rows*0.625);
		ptss[3] = referenceLine_ptss[3] = cv::Point(currImg.cols*0.9296, currImg.rows*0.8666);
	}
#endif
	const cv::Point *polygons[2] = { ptss, };
	int ntps[2] = { 4, };
	//ReferenceLine Mask
	cv::polylines(currImg, polygons, ntps, 1, false, cv::Scalar(0, 255, 0));
}

void DetectionProcessing::convertDataToProtocolMsg(DetPeople_INFO *det, int detCnt, int day_night)
{
	unsigned char crcData = NULL;
	DATA_BITFIELD *dt = &(dataCommunication_ptr->prtInfo.data);
	unsigned char* dataField;
	DetPeople_INFO temp;

	if (detCnt == 0)
	{
		dataCommunication_ptr->prtInfo.sync = 0x80;
		memset(&dataCommunication_ptr->prtInfo.data, 0x00, sizeof(dataCommunication_ptr->prtInfo.data));
		dataCommunication_ptr->prtInfo.data.command = 0x03;		//command field : 0x03	,	extra field : 0x00
		dataCommunication_ptr->prtInfo.data.day_night = day_night;

		dataField = (unsigned char*)dt;
		crcData = dataCommunication_ptr->crcCalculate8_SAE_J1850(dataField, 11);		//data field만 계산, 총 11byte
		dataCommunication_ptr->prtInfo.crc = crcData;
	}
	else if (detCnt == 1)
	{
		dataCommunication_ptr->prtInfo.sync = 0x80;
		memset(&dataCommunication_ptr->prtInfo.data, 0x00, sizeof(dataCommunication_ptr->prtInfo.data));
		dataCommunication_ptr->prtInfo.data.command = 0x03;
		dataCommunication_ptr->prtInfo.data.day_night = day_night;

		dataCommunication_ptr->prtInfo.data.detection_cnt = 0x01;
		dataCommunication_ptr->prtInfo.data.l_position1 = det[0]._prt_position;
		dataCommunication_ptr->prtInfo.data.l_risk1 = det[0]._prt_risk;
		dataCommunication_ptr->prtInfo.data.l_direction1 = det[0]._prt_direction;
		dataCommunication_ptr->prtInfo.data.distance1 = det[0]._prt_distance;
		dataCommunication_ptr->prtInfo.data.riskrate1 = det[0]._prt_riskRate;

		dataField = (unsigned char*)dt;
		crcData = dataCommunication_ptr->crcCalculate8_SAE_J1850(dataField, 11);
		dataCommunication_ptr->prtInfo.crc = crcData;
	}
	else if (detCnt == 2)
	{
		dataCommunication_ptr->prtInfo.sync = 0x80;
		memset(&dataCommunication_ptr->prtInfo.data, 0x00, sizeof(dataCommunication_ptr->prtInfo.data));
		dataCommunication_ptr->prtInfo.data.command = 0x03;
		dataCommunication_ptr->prtInfo.data.day_night = day_night;

		dataCommunication_ptr->prtInfo.data.detection_cnt = 0x02;

		if (det[0].distance > det[1].distance)
		{
			temp = det[0];
			det[0] = det[1];
			det[1] = temp;
		}
		dataCommunication_ptr->prtInfo.data.l_position1 = det[0]._prt_position;
		dataCommunication_ptr->prtInfo.data.l_risk1 = det[0]._prt_risk;
		dataCommunication_ptr->prtInfo.data.l_direction1 = det[0]._prt_direction;
		dataCommunication_ptr->prtInfo.data.distance1 = det[0]._prt_distance;
		dataCommunication_ptr->prtInfo.data.riskrate1 = det[0]._prt_riskRate;

		dataCommunication_ptr->prtInfo.data.l_position2 = det[1]._prt_position;
		dataCommunication_ptr->prtInfo.data.l_risk2 = det[1]._prt_risk;
		dataCommunication_ptr->prtInfo.data.l_direction2 = det[1]._prt_direction;
		dataCommunication_ptr->prtInfo.data.distance2 = det[1]._prt_distance;
		dataCommunication_ptr->prtInfo.data.riskrate2 = det[1]._prt_riskRate;

		dataField = (unsigned char*)dt;
		crcData = dataCommunication_ptr->crcCalculate8_SAE_J1850(dataField, 11);
		dataCommunication_ptr->prtInfo.crc = crcData;
	}
	// 검출 보행자가 3명이상일 때
	else
	{
		dataCommunication_ptr->prtInfo.sync = 0x80;
		memset(&dataCommunication_ptr->prtInfo.data, 0x00, sizeof(dataCommunication_ptr->prtInfo.data));
		dataCommunication_ptr->prtInfo.data.command = 0x03;
		dataCommunication_ptr->prtInfo.data.day_night = day_night;

		dataCommunication_ptr->prtInfo.data.detection_cnt = detCnt;
		for (int i = detCnt - 1; i >= 0; i--)
		{
			for (int j = 0; j < i; j++)
			{
				if (det[j]._prt_distance > det[j + 1]._prt_distance)
				{
					temp = det[j];
					det[j] = det[j + 1];
					det[j + 1] = temp;
				}
			}
		}
		dataCommunication_ptr->prtInfo.data.l_position1 = det[0]._prt_position;
		dataCommunication_ptr->prtInfo.data.l_risk1 = det[0]._prt_risk;
		dataCommunication_ptr->prtInfo.data.l_direction1 = det[0]._prt_direction;
		dataCommunication_ptr->prtInfo.data.distance1 = det[0]._prt_distance;
		dataCommunication_ptr->prtInfo.data.riskrate1 = det[0]._prt_riskRate;

		dataCommunication_ptr->prtInfo.data.l_position2 = det[1]._prt_position;
		dataCommunication_ptr->prtInfo.data.l_risk2 = det[1]._prt_risk;
		dataCommunication_ptr->prtInfo.data.l_direction2 = det[1]._prt_direction;
		dataCommunication_ptr->prtInfo.data.distance2 = det[1]._prt_distance;
		dataCommunication_ptr->prtInfo.data.riskrate2 = det[1]._prt_riskRate;

		dataCommunication_ptr->prtInfo.data.l_position3 = det[2]._prt_position;
		dataCommunication_ptr->prtInfo.data.l_risk3 = det[2]._prt_risk;
		dataCommunication_ptr->prtInfo.data.l_direction3 = det[1]._prt_direction;
		dataCommunication_ptr->prtInfo.data.distance3 = det[2]._prt_distance;
		dataCommunication_ptr->prtInfo.data.riskrate3 = det[2]._prt_riskRate;

		dataField = (unsigned char*)dt;
		crcData = dataCommunication_ptr->crcCalculate8_SAE_J1850(dataField, 11);
		dataCommunication_ptr->prtInfo.crc = crcData;
	}

#ifdef SIGNAL_INFO_SAVE
	std::string current_file_name = filename;
	current_file_name.pop_back();
	current_file_name.pop_back();
	current_file_name.pop_back();
	current_file_name = current_file_name + "txt";
	FILE *fp = fopen(current_file_name.c_str(), "a+");
	if (bFlag == false)
	{
		fprintf(fp, "%6s\t%6s\t%6s\t%6s\t%6s\n", "FrameNo", "Position", "Risk", "Distance", "Direction");
		bFlag = true;
	}
	
	std::sort(det, det + detCnt, cmp_Distance);
	std::cout << std::endl;
	if (detCnt > 2)
	{
		std::cout.precision(6);
		for (int a = 0; a < 3; a++)
		{
			std::cout << m_nFrameCount << "\t";
			std::cout << det[a]._prt_position << "\t" <<
				det[a]._prt_risk << "\t" <<det[a]._prt_distance<<"\t" <<det[a]._prt_direction << std::endl;
			if (bFlag == true)
			{
				fprintf(fp, "%d\t%d\t%d\t%d\t%d\n", m_nFrameCount, det[a]._prt_position, det[a]._prt_risk, det[a]._prt_distance,
					det[a]._prt_direction);
			}
		}
		
	}
	else if(1<= detCnt && detCnt <= 2)
	{
		std::cout.precision(6);
		for (int a = 0; a < detCnt; a++)
		{
			std::cout << m_nFrameCount << "\t";
			std::cout << det[a]._prt_position << "\t" <<
				det[a]._prt_risk << "\t" << det[a]._prt_distance<<"\t"<<det[a]._prt_direction << std::endl;
			if (bFlag == true)
			{
				fprintf(fp, "%d\t%d\t%d\t%d\t%d\n", m_nFrameCount, det[a]._prt_position, det[a]._prt_risk, det[a]._prt_distance,
					det[a]._prt_direction);
			}
		}
	}
	else
	{
		std::cout << m_nFrameCount << "\t";
		std::cout << detCnt<< "\t" <<
			detCnt << "\t" << detCnt << "\t" << detCnt << std::endl;
		if (bFlag == true)
			fprintf(fp, "%d\t%d\t%d\t%d\t%d\n", m_nFrameCount, 0,0,0,0);
	}
	fclose(fp);
#endif
}

void DetectionProcessing::draw_boxes_validation(cv::Mat mat_img, truth_box* all_truth_box,
	std::vector<bbox_t> result_vec, int num_labels, FILE *fp, int frame_no, int *current_TP, int *current_FP)
{
	int current_det_fps = -1;
	int current_cap_fps = -1;

	int all_detected = 0;
	int tp_cnt = 0;
	Display_StaticReferenceMask_Validation(mat_img);
	set_StaticReferenceMask();
	if (result_vec.size() == 0)
	{
		fprintf(stderr, "%7d\t%5d\t%2d\t%12d\t%13d\n", frame_no, 0, 0, 0, 0);
		fprintf(fp, "%7d\t%5d\t%2d\t%12d\t%13d\n", frame_no, 0, 0, 0, 0);
		*current_TP += 0;
		*current_FP += 0;
		return;
	}
	//검출된 모든 보행자의 정보 저장

	if (result_vec.size() != 0)
	{
		tp_box all_tp_boxes[15] = { 0, };
		det_box all_det_boxes[15] = { 0, };

		//det_box* all_det_box = new det_box[result_vec.size()];
		//det_box* all_det_box = (det_box*)malloc(sizeof(det_box)*result_vec.size());
		//tp_box* all_tp_box = new tp_box[result_vec.size()];
		//tp_box* all_tp_box = (tp_box*)malloc(sizeof(tp_box)*result_vec.size());

		/*for (auto i = 0; i < result_vec.size(); i++)
		{
		all_det_box[i].bottom = 0.0;
		all_det_box[i].top = 0.0;
		all_det_box[i].right = 0.0;
		all_det_box[i].left = 0.0;
		all_det_box[i].spc_label = 0;
		all_det_box[i].class_id = 0;
		all_det_box[i].distance = 0.0;
		all_det_box[i].h = 0.0;
		all_det_box[i].w = 0.0;
		all_det_box[i].x = 0.0;
		all_det_box[i].y = 0.0;

		all_tp_box[i].class_id = 0;
		all_tp_box[i].gt_spc_label = 0;
		all_tp_box[i].st_spc_label = 0;
		all_tp_box[i].h = 0.0;
		all_tp_box[i].w = 0.0;
		all_tp_box[i].x = 0.0;
		all_tp_box[i].y = 0.0;
		}*/

		detection_sudden_crossing_Validation(mat_img, result_vec, all_det_boxes);
		std::sort(all_truth_box, all_truth_box + num_labels, cmp_truth_boxes);
		std::sort(all_det_boxes, all_det_boxes + result_vec.size(), cmp_det_boxes);
		detection_tp_check(all_truth_box, all_det_boxes, all_tp_boxes, num_labels, result_vec.size(), &tp_cnt);
		//for (auto &i : result_vec) {
		//
		//	//cv::Scalar color = obj_id_to_color(i.obj_id);
		//	//cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), cv::Scalar(50, 255, 0), 2);
		//	//if (obj_names.size() > i.obj_id) {
		//	//	std::string obj_name = obj_names[i.obj_id];
		//	//	if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
		//	//	cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
		//	//	int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);

		//	//	//cv::rectangle(mat_img, cv::Point2f(max((int)i.x - 1, 0), max((int)i.y - 30, 0)),
		//	//	//	cv::Point2f(min((int)i.x + max_width, mat_img.cols - 1), min((int)i.y, mat_img.rows - 1)),
		//	//	//	color, CV_FILLED, 8, 0);
		//	//}
		//}
		//for (int i = 0; i < all_detected; i++)
		if (tp_cnt == 0)
		{
			fprintf(stderr, "%7d\t%5d\t%2d\t%12d\t%13d\n", frame_no, 0, 0, 0, 0);
			fprintf(fp, "%7d\t%5d\t%2d\t%12d\t%13d\n", frame_no, 0, 0, 0, 0);
			*current_TP += 0;
			*current_FP += 0;
		}
		else
		{
			for (int i = 0; i < tp_cnt; i++)
			{
				int true_positive = 0;
				int false_positive = 0;
				int x, y, w, h;
				x = all_tp_boxes[i].x;
				y = all_tp_boxes[i].y;
				w = all_tp_boxes[i].w;
				h = all_tp_boxes[i].h;
				if (all_tp_boxes[i].st_spc_label == 0)
				{
				cv::rectangle(mat_img, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
				cv::putText(mat_img, "ST : Normal", cv::Point(x + w + 20, y + (h / 5 * 4)),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(40, 177, 255), 2);
				}
				else if (all_tp_boxes[i].st_spc_label == 1)
				{
				cv::rectangle(mat_img, cv::Rect(x, y, w, h), cv::Scalar(40, 177, 255), 2);
				cv::putText(mat_img, "ST : Caution", cv::Point(x + w + 20, y + (h / 5 * 4)),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(40, 177, 255), 2);
				}
				else
				{
				cv::rectangle(mat_img, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);
				cv::putText(mat_img, "ST : Warning", cv::Point(x + w + 20, y + (h / 5 * 4)),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(40, 177, 255), 2);
				}
				if (all_tp_boxes[i].gt_spc_label == 0)
				cv::putText(mat_img, "GT : Normal", cv::Point(x + w + 20, y + (h)),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 2);
				else if (all_tp_boxes[i].gt_spc_label == 1)
				cv::putText(mat_img, "GT : Caution", cv::Point(x + w + 20, y + (h)),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 2);
				else
				cv::putText(mat_img, "GT : Warning", cv::Point(x + w + 20, y + (h)),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 2);
				if (all_tp_boxes[i].gt_spc_label == all_tp_boxes[i].st_spc_label)
				true_positive = 1;
				else
				false_positive = 1;
				/*if (all_tp_boxes[i].st_spc_label == 0)
				{
					cv::rectangle(mat_img, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
					cv::putText(mat_img, "Normal", cv::Point(x + w + 20, y + (h / 5 * 4)),
						cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);
				}
				else if (all_tp_boxes[i].st_spc_label == 1)
				{
					cv::rectangle(mat_img, cv::Rect(x, y, w, h), cv::Scalar(40, 177, 255), 2);
					cv::putText(mat_img, "Caution", cv::Point(x + w + 20, y + (h / 5 * 4)),
						cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(40, 177, 255), 2);
				}
				else
				{
					cv::rectangle(mat_img, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);
					cv::putText(mat_img, "Warning", cv::Point(x + w + 20, y + (h / 5 * 4)),
						cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 3);
				}*/

				if (all_tp_boxes[i].gt_spc_label == all_tp_boxes[i].st_spc_label)
					true_positive = 1;
				else
					false_positive = 1;
				if (all_tp_boxes[i].gt_spc_label == 0)
					normal_gt_count++;
				else if (all_tp_boxes[i].gt_spc_label == 1)
					caution_gt_count++;
				else if (all_tp_boxes[i].gt_spc_label == 2)
					warning_gt_count++;
				
				if (all_tp_boxes[i].st_spc_label == 0 && true_positive == 1)
					normal_st_count++;
				else if (all_tp_boxes[i].st_spc_label == 1 && true_positive == 1)
					caution_st_count++;
				else if (all_tp_boxes[i].st_spc_label == 2 && true_positive == 1)
					warning_st_count++;
				fprintf(stderr, "%7d\t%5d\t%2d\t%12d\t%13d\n", frame_no, all_tp_boxes[i].gt_spc_label,
					all_tp_boxes[i].st_spc_label, true_positive, false_positive);
				fprintf(fp, "%7d\t%5d\t%2d\t%12d\t%13d\n", frame_no, all_tp_boxes[i].gt_spc_label,
					all_tp_boxes[i].st_spc_label, true_positive, false_positive);
				*current_TP += true_positive;
				*current_FP += false_positive;
			}
		}

		if (current_det_fps >= 0 && current_cap_fps >= 0) {
			std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) +
				"   FPS capture: " + std::to_string(current_cap_fps);
			putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2,
				cv::Scalar(50, 255, 0), 2);
		}
		//delete[] all_det_box;
		//free(all_det_box);
		////delete[] all_tp_box;
		//free(all_tp_box);
	}
}

void DetectionProcessing::detection_sudden_crossing_Validation(cv::Mat currImg, std::vector<bbox_t> result_vec,
	det_box *all_det_boxes)
{
	int k, f;

	double dDecision_OverlappedRate = 0.0;
	double dDecision_MovementDir = 0.0;
	double dDecision_MovementMag = 0.0;
	double dDecision_Distance = 0.0;
	double dDecision_Overlapped_prob = 0.0;
	double dDecision_MovementDir_prob = 0.0;
	double dDecision_MovementMag_prob = 0.0;
	double dDecision_Distance_prob = 0.0;
	double dExceptionalRiskValue = 0.0;
	int nDistanceFromCam = 100;                 // 보행자 윈도우 ~ 카메라간 거리 (2015-01-19)
	int nWindowPositionIn3part = 0;             // 보행자 윈도우의 Reference Line 기준으로 위치 (0:중심, 1:Left RL 겹침 위치, 2:Right RL 겹침 위치)
	double dResult_prob = 0;

	int nTmpCount = 0;
	int x, y;
	int detPeopleCount = 0;

	for (auto &i : result_vec)
		detPeopleCount++;

	DetPeople_INFO *detectedPeopleInfo;
	if (detPeopleCount > 0)
		detectedPeopleInfo = new DetPeople_INFO[detPeopleCount];
	else
	{
		detectedPeopleInfo = new DetPeople_INFO[1];
		detectedPeopleInfo[0] = {};
	}

	//YOLO에 의해 검출된 보행자 윈도우 정보를 담을 구조체
	//DetPeople_INFO *detectedPeopleInfo = new DetPeople_INFO[detPeopleCount];
	int m = 0;
	for (auto &i : result_vec)
	{
		double people_height = i.h;
		double people_distance = REAL_PEOPLE_HEIGHT * FOCAL_LENGTH / people_height;

		// 보행자 횡방향 : 좌, 우, 센터
		int rowDirection = IMG_SIZE_WIDTH / 3;
		int centerX = i.x + i.w / 2;
		cv::Point2f rowDirection_position;
		rowDirection_position.x = i.x + i.w - 30;
		rowDirection_position.y = i.y + i.h + 55;
		int pedestrian_Position = 0;

		detectedPeopleInfo[m].det_id = m;

		///////////////영상내에서 보행자가 위치한 영역/////////////////////////////////

		if (0 < centerX && centerX < IMG_SIZE_WIDTH * 0.4)                              // Left
			pedestrian_Position = 0;
		else if (IMG_SIZE_WIDTH * 0.4 < centerX && centerX < IMG_SIZE_WIDTH * 0.6)      // Center
			pedestrian_Position = 1;
		else if (IMG_SIZE_WIDTH*0.6 < centerX && centerX < IMG_SIZE_WIDTH)              // Right
			pedestrian_Position = 2;

		all_det_boxes[m].x = detectedPeopleInfo[m].start_x = i.x;
		all_det_boxes[m].y = detectedPeopleInfo[m].start_y = i.y;
		detectedPeopleInfo[m].end_x = i.x + i.w;
		detectedPeopleInfo[m].end_y = i.y + i.h;
		detectedPeopleInfo[m].obj_class = i.obj_id;
		all_det_boxes[m].w = i.w;
		all_det_boxes[m].h = i.h;
		all_det_boxes[m].distance = detectedPeopleInfo[m].distance = people_distance;
		m++;
	}

	//YOLO에 의해 검출된 보행자 윈도우중 1개의 윈도우 정보
	DetPeople_INFO current_winInfo;

	for (k = 0; k < result_vec.size(); k++)
	{
		current_winInfo.start_x = detectedPeopleInfo[k].start_x;
		current_winInfo.start_y = detectedPeopleInfo[k].start_y;
		current_winInfo.end_x = detectedPeopleInfo[k].end_x;
		current_winInfo.end_y = detectedPeopleInfo[k].end_y;
		current_winInfo.det_id = detectedPeopleInfo[k].det_id;
		int ped_orient = detectedPeopleInfo[k].obj_class;
		int ped_position;
		int ped_coord_x = current_winInfo.start_x + (current_winInfo.end_x - current_winInfo.start_x) / 2;

		if (currImg.cols*0.3671 <= ped_coord_x && ped_coord_x <= currImg.cols*0.5156)
			ped_position = 0;
		else if (0 <= ped_coord_x && ped_coord_x < currImg.cols*0.3671)
			ped_position = 1;
		else
			ped_position = 2;
		double ped_distance = detectedPeopleInfo[k].distance;

		dDecision_OverlappedRate = 0.0;
		// 조건 (1) 체크 : 레퍼런스 라인과 보행자 윈도우간의 겹칩율
		dDecision_OverlappedRate = detectionSuddenCrossing_overlappedAreaRate(current_winInfo.start_x, current_winInfo.end_x,
			current_winInfo.start_y, current_winInfo.end_y, nWindowPositionIn3part);

		detectionSuddenCrossing_movement(currImg, ped_orient, ped_position, ped_distance,
			dDecision_MovementDir, dDecision_MovementMag);

		//거리가 가까울수록 높은 가중치 값
		dDecision_Distance_prob = 1.0 - detectedPeopleInfo[k].distance / MAX_DISTANCE;
		//dDecision_Distance_prob = detectedPeopleInfo[k].distance / MAX_DISTANCE;
		// 겹침비율, 움직임방향, 움직임크기와 가우시안 분포를 사용한 최종 보행자 검출
		dDecision_Distance_prob =
			detectionGaussianProbabilityDensityFunction(1 - dDecision_Distance_prob, 0.0, 0.5);
		//test//
		//dDecision_OverlappedRate = (dDecision_OverlappedRate *detectedPeopleInfo[k].distance/ MAX_DISTANCE);
		dDecision_Overlapped_prob =
			detectionGaussianProbabilityDensityFunction(1.0 - dDecision_OverlappedRate, 0.0, 0.6);
		//detectionGaussianProbabilityDensityFunction(dDecision_OverlappedRate, 0.0, 0.6);
		dDecision_MovementDir_prob =
			detectionGaussianProbabilityDensityFunction(1.0 - dDecision_MovementDir, 0.0, 0.5);
		dDecision_MovementMag_prob =
			detectionGaussianProbabilityDensityFunction(1.0 - dDecision_MovementMag, 0.0, 0.5);
		//dResult_prob = dDecision_Distance_prob*0.15 + dDecision_Overlapped_prob*0.6 + dDecision_MovementDir_prob*0.2
		//  + dDecision_MovementMag_prob*0.1 + dExceptionalRiskValue;       // 2015-01-19   예외적 위험 상황에 대한 가중치 부여
		//dResult_prob = dDecision_Distance_prob*0.25 + dDecision_Overlapped_prob*0.65 + dDecision_MovementDir_prob*0.1;

		dResult_prob = dDecision_Distance_prob*0.3 + dDecision_Overlapped_prob*0.6 + dDecision_MovementDir_prob*0.1;
		if (detectedPeopleInfo[k].distance <= 5.0)
			dResult_prob = (dDecision_Distance_prob - 0.4)*0.05 + dDecision_Overlapped_prob*0.9 + dDecision_MovementDir_prob*0.05;
		else if (5.0 <= detectedPeopleInfo[k].distance && detectedPeopleInfo[k].distance <= 10.0)
			dResult_prob = (dDecision_Distance_prob - 0.4)*0.6 + dDecision_Overlapped_prob*0.3 + dDecision_MovementDir_prob*0.1;
		else
			dResult_prob = (dDecision_Distance_prob - 0.4)*0.7 + dDecision_Overlapped_prob*0.2 + dDecision_MovementDir_prob*0.1;
		//ReferenceLine에 전혀 겹치지 않는경우 항상 Normal
		if (dDecision_OverlappedRate < 0.36)
		{
			int box_x, box_y, box_w, box_h;

			box_x = detectedPeopleInfo[k].start_x;
			box_y = detectedPeopleInfo[k].start_y;
			box_w = detectedPeopleInfo[k].end_x - detectedPeopleInfo[k].start_x;
			box_h = detectedPeopleInfo[k].end_y - detectedPeopleInfo[k].start_y;

			int height = (current_winInfo.end_y - current_winInfo.start_y) / 5 * 4;
			/*cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(0, 255, 0), 2);

			putText(currImg, "ST : Normal", cv::Point(box_x + box_w + 20, box_y+(box_h/5*4)),
			cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 255, 0), 2);*/
			all_det_boxes[k].spc_label = 0;
		}
		else
		{
			double dis = dDecision_Distance_prob *0.25;
			double overl = dDecision_Overlapped_prob *0.5;

			if (dResult_prob > 1.0)
				dResult_prob = 1.0;

			int box_x, box_y, box_w, box_h;

			box_x = all_det_boxes[k].x;
			box_y = all_det_boxes[k].y;
			box_w = all_det_boxes[k].w;
			box_h = all_det_boxes[k].h;

			////위험도 스코어 표시
			//putText(currImg, std::to_string(dResult_prob), cv::Point(box_x + box_w - 30, box_y - 30),
			//  cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 1);

			//Normal
			//if (dResult_prob < 0.56)
			if (dResult_prob < 0.41)
				//std::cout << "result_prob : " << dResult_prob << std::endl;
				//if (dResult_prob < 0.4)
			{
				/*cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(0, 255, 0), 2);
				putText(currImg, "ST : Normal", cv::Point(box_x + box_w + 20, box_y + (box_h / 5 * 4)),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 255, 0), 2);*/
				all_det_boxes[k].spc_label = 0;
			}
			//Caution
			else if (0.41 < dResult_prob && dResult_prob < 0.695)
				//else if (0.4 < dResult_prob && dResult_prob < 0.6)
			{
				/*cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(40, 127, 255), 2);
				putText(currImg, "ST : Caution", cv::Point(box_x + box_w + 20, box_y + (box_h / 5 * 4)),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(40, 127, 255), 2);*/
				all_det_boxes[k].spc_label = 1;
			}
			//Warning
			else
			{
				/*cv::rectangle(currImg, cv::Rect(box_x, box_y, box_w, box_h), cv::Scalar(0, 0, 255), 2);
				putText(currImg, "ST : Warning", cv::Point(box_x + box_w + 20, box_y + (box_h / 5 * 4)),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 2);*/
				all_det_boxes[k].spc_label = 2;
			}
		}
	}
	//결과 이미지 저장
	//char fileName[5];
	//char filePath[256] = "ResultImage/";
	//char saveImage[256];
	//sprintf(saveImage, "%s%d.jpg", filePath, m_nCurrnetFrameNum);
	//cv::imwrite(saveImage, currImg);
	//convertDataToProtocolMsg(detectedPeopleInfo, detPeopleCount, 0);      //0 :주간, 1: 야간

	//delete[] detectedPeopleInfo;
}

void DetectionProcessing::Display_StaticReferenceMask_Validation(cv::Mat currImg)
{
	cv::Point ptss[4];
	//Day time
	if (day_or_night == 0)
	{
		ptss[0] = referenceLine_ptss[0] = cv::Point(currImg.cols*0.0006, currImg.rows*0.8194);
		ptss[1] = referenceLine_ptss[1] = cv::Point(currImg.cols*0.3671, currImg.rows*0.4861);
		ptss[2] = referenceLine_ptss[2] = cv::Point(currImg.cols*0.5156, currImg.rows*0.4861);
		ptss[3] = referenceLine_ptss[3] = cv::Point(currImg.cols*0.9296, currImg.rows*0.9027);
	}
	//Night time
	else
	{
		ptss[0] = referenceLine_ptss[0] = cv::Point(currImg.cols*0.0006, currImg.rows*0.8666);
		ptss[1] = referenceLine_ptss[1] = cv::Point(currImg.cols*0.3612, currImg.rows*0.625);
		ptss[2] = referenceLine_ptss[2] = cv::Point(currImg.cols*0.5159, currImg.rows*0.625);
		ptss[3] = referenceLine_ptss[3] = cv::Point(currImg.cols*0.9296, currImg.rows*0.8666);
	}

	const cv::Point *polygons[2] = { ptss, };
	int ntps[2] = { 4, };
	//ReferenceLine Mask
	cv::polylines(currImg, polygons, ntps, 1, false, cv::Scalar(0, 255, 0));
}

void DetectionProcessing::find_replace(char* str, char* orig, char* rep, char* output)
{
	char buffer[4096] = { 0 };
	char *p;

	sprintf(buffer, "%s", str);
	if (!(p = strstr(buffer, orig))) {  // Is 'orig' even in 'str'?
		sprintf(output, "%s", str);
		return;
	}

	*p = '\0';

	sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
}

void DetectionProcessing::convert_gt_coordinate(cv::Mat currImg, truth_box* ground_truth, int num_labels)
{
	for (auto i = 0; i < num_labels; i++)
	{
		ground_truth[i].x = (currImg.cols * ground_truth[i].x) - (currImg.cols * ground_truth[i].w / 2);
		ground_truth[i].y = (currImg.rows * ground_truth[i].y) - (currImg.rows * ground_truth[i].h / 2);
		ground_truth[i].w = currImg.cols * ground_truth[i].w;
		ground_truth[i].h = currImg.rows * ground_truth[i].h;
	}
}

void DetectionProcessing::detection_tp_check(truth_box* t_box, det_box* d_box, tp_box* all_tp_box, int num_labels,
	int all_det_cnt, int *tp_cnt)
{
	int idx = 0;
	for (auto j = 0; j < num_labels; j++)
	{
		float iou = 0.0;
		for (auto i = 0; i < all_det_cnt; i++)
		{
			iou = box_iou(t_box[j], d_box[i]);
			if (iou > 0.4)
			{
				all_tp_box[idx].x = d_box[i].x;
				all_tp_box[idx].y = d_box[i].y;
				all_tp_box[idx].w = d_box[i].w;
				all_tp_box[idx].h = d_box[i].h;
				all_tp_box[idx].st_spc_label = d_box[i].spc_label;
				all_tp_box[idx].gt_spc_label = t_box[j].spc_label;
				(*tp_cnt)++;
				idx++;
			}
		}
	}
}

void DetectionProcessing::draw_box_test(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int current_det_fps = -1;
	int current_cap_fps = -1;
	for (auto &i : result_vec) {
		//cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), cv::Scalar(0, 255, 0), 2);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			/*int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0)),
			cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
			cv::Scalar(0,0,255), CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);*/
		}
	}
	if (current_det_fps >= 0 && current_cap_fps >= 0) {
		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
	}
}

int DetectionProcessing::decide_day_and_night(cv::Mat mat_img)
{
	double total_pixel = 0;
	double sum_pixel_value = 0;
	double average_pixel_value = 0;

	for (auto y = 0; y < IMG_SIZE_HEIGHT*0.15; ++y)
	{
		uchar* pointer_input = mat_img.ptr<uchar>(y);
		double one_pixel = 0;
		for (auto x = 0; x < IMG_SIZE_WIDTH; ++x)
		{
			uchar pixel_b = pointer_input[x * 3 + 0];
			uchar pixel_g = pointer_input[x * 3 + 1];
			uchar pixel_r = pointer_input[x * 3 + 2];
			total_pixel++;
			one_pixel = (pixel_b + pixel_g + pixel_r) / 3.0;
			sum_pixel_value += one_pixel;
		}
	}
	average_pixel_value = sum_pixel_value / total_pixel;

	//Daytime
	if (average_pixel_value > 50)
		return 0;
	//Nighttime
	else
		return 1;
}

void DetectionProcessing::set_StaticReferenceMask()
{
	get_ReferenceLine_Mask();
	cvFillConvexPoly(m_ref_mask_img, referenceLine_ptss, 4, cvScalar(255, 255, 255));
}

void DetectionProcessing::orientation_display(cv::Mat curr_img, int ped_orientation, DetPeople_INFO det_info)
{
	//double degree[8] = { -90, 90, -135, 135, 180, -45, 45, 0 };
	double degree[8] = { 0,45,90,135,180,-135,-90,-45 };
	double rad[8];
	for (int i = 0; i < 8; i++)
		rad[i] = degree[i] * PI / 180;

	int radius = 50;

	double coord_x = radius*cos(rad[ped_orientation]);
	double coord_y = radius*sin(rad[ped_orientation]);

	double center_x = (det_info.end_x - det_info.start_x) / 2 + det_info.start_x;
	double center_y = (det_info.end_y - det_info.start_y) / 2 + det_info.start_y;

	//cv::line(orient_img, cv::Point(35, 25), cv::Point(35 +coord_x, 25 + coord_y), cv::Scalar(0, 0, 0), 1.5);
#if RESULT_DISPLAY
	cv::arrowedLine(curr_img, cv::Point(center_x, center_y), cv::Point(center_x + coord_x, center_y + coord_y), cv::Scalar(0, 242, 255), 3.5);
#endif
}

void DetectionProcessing::get_ReferenceLine_Mask()
{
	for (auto i = 0; i < IMG_SIZE_HEIGHT; i++)
	{
		for (auto j = 0; j < IMG_SIZE_WIDTH; j++)
		{
			m_mask[i][j] = m_ref_mask_img->imageData[i*m_ref_mask_img->widthStep + j];
		}
	}
}