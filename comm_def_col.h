#ifndef _COMM_DEF_H
#define _COMM_DEF_H

//#include <Windows.h>  
#include <intrin.h>
#include <basetsd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <process.h>
#include <cstring>
#include <cstringt.h>
//#include <afx.h>
#include <sys/types.h>
#include <atlimage.h>
using namespace std;

#include "../vicImgPro/vicImgPro.h"
#include "../vicImgPro/cv_lib.h"
#ifdef _DEBUG
#pragma comment(lib, "../Debug/vicImgPro.lib") //添加lib文件引用 
#else
#pragma comment(lib, "../Release/vicImgPro.lib") //添加lib文件引用
#endif

#ifdef _DEBUG
#define SAVE_FLAG				(1)
#else
#define SAVE_FLAG				(0)
#endif

#endif