//This code is just to get familarise with the opencv function and the calculation of hog descriptor and visualizing the same
//NOTE: Change the data path according to the requirement.

#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <iterator>

// for opencv functionality
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include <opencv2/opencv.hpp>

bool visualize_progress = true;

//Function declaration
void visualizeHOG(cv::Mat & img, std::vector<float> &feats, cv::HOGDescriptor & hog_detector, int scale_factor);

int main()
{
//    Reading the original image and creating different images using basic image processing in openCV

	cv::String image_path = std::string("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/TDCV/TDCV/obj1000.jpg");

	cv::Mat original_image;
	try
	{
		original_image = cv::imread(image_path, 1);

		if (visualize_progress)
		{
			std::cout << original_image.size() << std::endl;
			cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Image", original_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during reading the original image: " << ex.what() << "\n\n";
	}


	// Converting the original image to grayscale
	cv::Mat grayscale_image;
	try
	{
		cv::cvtColor(original_image, grayscale_image, CV_RGB2GRAY);

		if (visualize_progress)
		{
			std::cout << grayscale_image.size() << std::endl;
			cv::namedWindow("Grayscale Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Grayscale Image", grayscale_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during converting the original image to grayscale: " << ex.what() << "\n\n";
	}


	// Expanding the original image
	cv::Mat expanded_image;
	try
	{
		cv::resize(original_image, expanded_image, cv::Size(), 2.0, 2.0);

		if (visualize_progress)
		{
			std::cout << expanded_image.size() << std::endl;
			cv::namedWindow("Expanded Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Expanded Image", expanded_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during expanding the original image: " << ex.what() << "\n\n";
	}


	// Compressing the original image
	cv::Mat compressed_image;
	try
	{
		cv::resize(original_image, compressed_image, cv::Size(), 0.5, 0.5);

		if (visualize_progress)
		{
			std::cout << compressed_image.size() << std::endl;
			cv::namedWindow("Compressed Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Compressed Image", compressed_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during compressing the original image: " << ex.what() << "\n\n";
	}


	// Rotating the original image
	cv::Mat rotated_image;
	try
	{
		cv::rotate(original_image, rotated_image, cv::ROTATE_90_CLOCKWISE);

		if (visualize_progress)
		{
			std::cout << rotated_image.size() << std::endl;
			cv::namedWindow("Rotated Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Rotated Image", rotated_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during rotating the original image: " << ex.what() << "\n\n";
	}


	// Flipping the original image
	cv::Mat flipped_image;
	try
	{
		cv::flip(original_image, flipped_image, 0);

		if (visualize_progress)
		{
			std::cout << flipped_image.size() << std::endl;
			cv::namedWindow("Flipped Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Flipped Image", flipped_image);
			cv::waitKey(3000);
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during flipping the original image: " << ex.what() << "\n\n";
	}

//  Calculating the HOG Descriptors of all different images created above
	
	cv::Rect region(0, 0, 124, 104);
	cv::Mat cropped_image = original_image(region);
	cv::Size cellsize(8, 8);
	cv::Size blocksize(16, 16);
	cv::Size stridesize(4, 4);
	cv::Size winsize(cropped_image.cols, cropped_image.rows);
	
	cv::HOGDescriptor hog_cropped_image(winsize, blocksize, stridesize, cellsize, 9);
	std::vector<float> descriptors_cropped_image;

	try
	{
		hog_cropped_image.compute(cropped_image,
								   descriptors_cropped_image,
								   cv::Size(0,0),
								   cv::Size(0, 0));
        visualizeHOG(cropped_image, descriptors_cropped_image,hog_cropped_image,1);
		cv::waitKey(3000);
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during calculating the HOG descriptor: " << ex.what() << "\n\n";
	}
	return 0;
}

//Visualizing the HOG descriptor (same code as provides), apart from green color and scale factor to 2.5
void visualizeHOG(cv::Mat & img, std::vector<float> &feats, cv::HOGDescriptor & hog_detector, int scale_factor) {

    cv::Mat visual_image;
    resize(img, visual_image, cv::Size(img.cols * scale_factor, img.rows * scale_factor));

    int n_bins = hog_detector.nbins;
    float rad_per_bin = 3.14 / (float) n_bins;
    cv::Size win_size = hog_detector.winSize;
    cv::Size cell_size = hog_detector.cellSize;
    cv::Size block_size = hog_detector.blockSize;
    cv::Size block_stride = hog_detector.blockStride;

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = win_size.width / cell_size.width;
    int cells_in_y_dir = win_size.height / cell_size.height;
    int n_cells = cells_in_x_dir * cells_in_y_dir;
    int cells_per_block = (block_size.width / cell_size.width) * (block_size.height / cell_size.height);

    int blocks_in_x_dir = (win_size.width - block_size.width) / block_stride.width + 1;
    int blocks_in_y_dir = (win_size.height - block_size.height) / block_stride.height + 1;
    int n_blocks = blocks_in_x_dir * blocks_in_y_dir;

    float ***gradientStrengths = new float **[cells_in_y_dir];
    int **cellUpdateCounter = new int *[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float *[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[n_bins];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < n_bins; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }


    // compute gradient strengths per cell
    int descriptorDataIdx = 0;


    for (int block_x = 0; block_x < blocks_in_x_dir; block_x++) {
        for (int block_y = 0; block_y < blocks_in_y_dir; block_y++) {
            int cell_start_x = block_x * block_stride.width / cell_size.width;
            int cell_start_y = block_y * block_stride.height / cell_size.height;

            for (int cell_id_x = cell_start_x;
                 cell_id_x < cell_start_x + block_size.width / cell_size.width; cell_id_x++)
                for (int cell_id_y = cell_start_y;
                     cell_id_y < cell_start_y + block_size.height / cell_size.height; cell_id_y++) {

                    for (int bin = 0; bin < n_bins; bin++) {
                        float val = feats.at(descriptorDataIdx++);
                        gradientStrengths[cell_id_y][cell_id_x][bin] += val;
                    }
                    cellUpdateCounter[cell_id_y][cell_id_x]++;
                }
        }
    }


    // compute average gradient strengths
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

            float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < n_bins; bin++) {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }


    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cell_size.width;
            int drawY = celly * cell_size.height;

            int mx = drawX + cell_size.width / 2;
            int my = drawY + cell_size.height / 2;

            rectangle(visual_image,
                      cv::Point(drawX * scale_factor, drawY * scale_factor),
                      cv::Point((drawX + cell_size.width) * scale_factor,
                                (drawY + cell_size.height) * scale_factor),
                      CV_RGB(100, 100, 100),
                      1);

            for (int bin = 0; bin < n_bins; bin++) {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                if (currentGradStrength == 0)
                    continue;

                float currRad = bin * rad_per_bin + rad_per_bin / 2;

                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float maxVecLen = cell_size.width / 2;
                //float scale = scale_factor / 5.0; // just a visual_imagealization scale,
                float scale = 2.5;
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visual_imagealization
                line(visual_image,
                     cv::Point(x1 * scale_factor, y1 * scale_factor),
                     cv::Point(x2 * scale_factor, y2 * scale_factor),
                     CV_RGB(0, 255, 0),
                     1);

            }

        }
    }


    for (int y = 0; y < cells_in_y_dir; y++) {
        for (int x = 0; x < cells_in_x_dir; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    cv::imshow("HOG vis", visual_image);
    //cv::waitKey(0);
    std::cout<<"reached";
    cv::imwrite("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/TDCV/TDCV/hog_vis.jpg", visual_image);

}
