#include "Validation.hpp"
#include <iostream>
//
truth_box* read_truth_boxes(char* filename, int *n)
{
	truth_box *boxes = static_cast<truth_box*>(calloc(1, sizeof(truth_box)));
	FILE* file = fopen(filename, "r");
	if (file == NULL)
	{
		std::cout << "File open failed" << std::endl;
		return NULL;
	}

	float x = 0, y = 0, w = 0, h = 0;
	int id = 0;
	int spc_label = 0;
	int count = 0;


	while (fscanf_s(file, "%d %f %f %f %f %d", &id, &x, &y, &w, &h, &spc_label) == 6)
	{
		boxes = static_cast<truth_box*>(realloc(boxes, (count + 1) * sizeof(truth_box)));
		boxes[count].class_id = id;
		boxes[count].x = x;
		boxes[count].y = y;
		boxes[count].h = h;
		boxes[count].w = w;
		boxes[count].left = x - w / 2;
		boxes[count].right = x + w / 2;
		boxes[count].top = y - h / 2;
		boxes[count].bottom = y + h / 2;
		boxes[count].spc_label = spc_label;
		++count;
	}
	fclose(file);
	*n = count;
	return boxes;
}
bool cmp_truth_boxes(const truth_box &box1, const truth_box &box2)
{
	if (box1.x < box2.x)
		return true;
	else if (box1.x == box2.x)
		return box1.y < box2.y;
	else
		return false;
}
bool cmp_det_boxes(const det_box &box1, const det_box &box2)
{
	if (box1.x < box2.x)
		return true;
	else if (box1.x == box2.x)
		return box1.y < box2.y;
	else
		return false;
}
float box_iou(truth_box t, det_box d)
{
	return box_intersection(t, d) / box_union(t, d);
}
float box_intersection(truth_box t, det_box d)
{
	float w = overlap(t.x, t.w, d.x, d.w);
	float h = overlap(t.y, t.h, d.y, d.h);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}
float box_union(truth_box t, det_box d)
{
	float i = box_intersection(t, d);
	float u = t.w * t.h + d.w*d.h - i;

	return u;
}

float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;

}