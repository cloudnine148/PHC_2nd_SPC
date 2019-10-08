#pragma once

typedef unsigned char BYTE;

#define RND(r) ((int)(r+0.5))
#define PI					3.141592653
#define NEGATIVE_TO_ZERO(a) ((a < 0) ? 0 : a)

// Default Image Size
#define IMG_SIZE_WIDTH							1280	
#define IMG_SIZE_HEIGHT							720

// Optical Flow�� Features ����Ʈ�� Grid ����
#define OPTICALfLOW_GRID_INTERVAL				10		// 2014-07-08 15->10

// detection ���� �� ��ŵ�� ������ ��
#define SKIP_FRAME					3//2
#define SKIP_FRAME_FOR_OPTICAL		3 //3

#define RESULT_FILE "./result.avi"