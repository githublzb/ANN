#include "comm_def_col.h"
//http://stackoverflow.com/questions/22470706/opencv-adaptivethreshold-versus-otsu-threshold-roi


int threshold_value = 0;
int threshold_type = 3;
int threshold_adaptive = 0;
int threshold_size = 3;

int const max_value = 255;
int const max_type = 4;
int const max_adaptive = 1;
int const max_size = 100;

int const max_BINARY_value = 255;

Mat src, src_gray, dst;
const char* window_name = "Threshold Demo";

const char* trackbar_adaptive = "Adaptive";
const char* trackbar_size = "Size";
const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

/// Function headers
void Threshold_Demo( int, void* );

/**
 * @function main
 */
int adaptive_thresh( int, string file_name)
{
  /// Load an image
  src = imread( file_name, 1 );

  /// Convert the image to Gray
  cvtColor( src, src_gray, COLOR_RGB2GRAY );

  /// Create a window to display results
  namedWindow( window_name, WINDOW_AUTOSIZE );

  /// Create Trackbar to choose type of Threshold
  createTrackbar( trackbar_type,
                  window_name, &threshold_type,
                  max_type, Threshold_Demo );

  createTrackbar( trackbar_value,
                  window_name, &threshold_value,
                  max_value, Threshold_Demo );

  createTrackbar( trackbar_adaptive,
                  window_name, &threshold_adaptive,
                  max_adaptive, Threshold_Demo );

  createTrackbar( trackbar_size,
                  window_name, &threshold_size,
                  max_size, Threshold_Demo );


  /// Call the function to initialize
  Threshold_Demo( 0, 0 );

  /// Wait until user finishes program
  for(;;)
    {
      int c;
      c = waitKey( 20 );
      if( (char)c == 27 )
    { break; }
    }
  return 0;
}


/**
 * @function Threshold_Demo
 */
void Threshold_Demo( int, void* )
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */
  if (threshold_adaptive)
  {
	int size = ((threshold_size ? threshold_size : 1) * 2 + 1);
	adaptiveThreshold(src_gray, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, size, threshold_value - 128);
  }
  else
	threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

  imshow( window_name, dst );
}
