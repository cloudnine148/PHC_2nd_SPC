#pragma once
#include <typeinfo>
//typedef struct
//{
//	int id;
//	float x, y, w, h;
//	float left, right, top, bottom;
//	int spc_label;
//}box_label_;
//
//typedef struct
//{
//	float x, y, w, h;
//}box_;
//box_label_ * boxes;
typedef struct
{
	int class_id;
	float x, y, w, h;
	float left, right, top, bottom;
	float distance;
	int spc_label;
	}truth_box;
typedef struct
{
	int class_id;
	float x, y, w, h;
	float left, right, top, bottom;
	float distance;
	int spc_label;
	 
}det_box;
bool cmp_truth_boxes(const truth_box &box1, const truth_box &box2);
bool cmp_det_boxes(const det_box &box1, const det_box &box2);

typedef struct
{
	int class_id;
	float x, y, w, h;
	int gt_spc_label;
	int st_spc_label;
}tp_box;
truth_box *read_truth_boxes(char* filename, int* n);

float box_iou(truth_box t, det_box d);
float box_intersection(truth_box t, det_box d);
float box_union(truth_box t, det_box d);
float overlap(float x1, float w1, float x2, float w2);