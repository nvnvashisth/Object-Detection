//In order to get the better result on the prediction, we need to have more data. The code below is to generate the augmented data.

#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <iterator>

#include  "opencv2/highgui/highgui.hpp"
#include  "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/opencv.hpp"

bool visualize_progress = true;
int tot_classes=6;
int num_train_files[] = {53, 81, 51, 290};
void data_augment();

int main()
{
    
   
    data_augment();
    
    return 0;
}

void data_augment()
{

cv::String imagepath("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/data/task3/");

int counter=-1;
cv::String folderName = "0" + std::to_string(3); //change this value for different class of object - 0 to 3 excepted

for (int j = 0; j < num_train_files[3]; j++) //change this value for different class of object - 0 to 3 excepted
   {
      cv::String imageName;

            if (j < 10)
                imageName = "000" + std::to_string(j) + ".jpg";
            else if (j < 100)
                imageName = "00" + std::to_string(j) + ".jpg";
            else
                imageName = "0" + std::to_string(j) + ".jpg";


	
	cv::String imgFile = imagepath + "train/" + folderName + "/" + imageName;

	// Reading the original image
	cv::Mat original_image;
	try
	{
		//original_image = cv::imread(image_path, 1);
		original_image = cv::imread(imgFile, CV_LOAD_IMAGE_COLOR);

		if (visualize_progress)
		{
			std::cout << original_image.size() << std::endl;
			cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Image", original_image);
			counter++;
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during reading the original image: " << ex.what() << "\n\n";
	}
	
	cv::String saveFile1 = imagepath + "trainaugmented/" + folderName + "/" + std::to_string(counter) + ".jpg";
	
	cv::imwrite( saveFile1, original_image );
	


	// Flipping the original image
	cv::Mat flipped1_image;
	try
	{
		cv::flip(original_image, flipped1_image, 0);

		if (visualize_progress)
		{
			std::cout << flipped1_image.size() << std::endl;
			cv::namedWindow("Flipped Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Flipped Image", flipped1_image);
			counter++;
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during flipping the original image: " << ex.what() << "\n\n";
	}
       
	cv::String saveFile2 = imagepath + "trainaugmented/" + folderName + "/" + std::to_string(counter)+ ".jpg";
	cv::imwrite( saveFile2, flipped1_image );



	//rotate the original image counter clockwise 90
	cv::Mat rotated2_image;
	try
	{		
		cv::rotate(original_image, rotated2_image, cv::ROTATE_90_COUNTERCLOCKWISE);

		if (visualize_progress)
		{
			std::cout << rotated2_image.size() << std::endl;
			cv::namedWindow("Rotated Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Rotated Image", rotated2_image);
			//cv::waitKey(1000);
			counter++;
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during rotating the original image: " << ex.what() << "\n\n";
	}

	cv::String saveFile3 = imagepath + "trainaugmented/" + folderName + "/" + std::to_string(counter)+ ".jpg";
	cv::imwrite( saveFile3, rotated2_image );
	


	//rotate the flipped(2) image clockwise 90
	cv::Mat rotated_flipped3_image;
	try
	{
		cv::rotate(flipped1_image, rotated_flipped3_image , cv::ROTATE_90_CLOCKWISE);

		if (visualize_progress)
		{
			std::cout << rotated_flipped3_image.size() << std::endl;
			cv::namedWindow("Rotated Flipped Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Rotated Flipped Image", rotated_flipped3_image);
			counter++;
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during flipping the original image: " << ex.what() << "\n\n";
	}

    cv::String saveFile4 = imagepath + "trainaugmented/" + folderName + "/" + std::to_string(counter)+ ".jpg";
	cv::imwrite( saveFile4, rotated_flipped3_image  );
	



	//Flipp on both axis 
	cv::Mat flipped4_image;
	try
	{
		cv::flip(original_image, flipped4_image, -1);

		if (visualize_progress)
		{
			std::cout << flipped4_image.size() << std::endl;
			cv::namedWindow(" both axis Flipped Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("both axis Flipped Image", flipped4_image);
			counter++;
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during flipping the original image: " << ex.what() << "\n\n";
	}
    
	cv::String saveFile5 = imagepath + "trainaugmented/" + folderName + "/" + std::to_string(counter)+ ".jpg";
	cv::imwrite( saveFile5, flipped4_image);




	//flipp on y axis 
	cv::Mat flipped5_image;
	try
	{
		cv::flip(original_image, flipped5_image, 1);

		if (visualize_progress)
		{
			std::cout << flipped5_image.size() << std::endl;
			cv::namedWindow(" both axis Flipped Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("both axis Flipped Image", flipped5_image);
			counter++;
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during flipping the original image: " << ex.what() << "\n\n";
	}

	cv::String saveFile6 = imagepath + "trainaugmented/" + folderName + "/" +std::to_string(counter)+ ".jpg";
	cv::imwrite( saveFile6,flipped5_image);


	//rotate the original image clockwise 90
		cv::Mat rotated6_image;
		try
		{		
			cv::rotate(original_image, rotated6_image, cv::ROTATE_90_CLOCKWISE);

			if (visualize_progress)
			{
				std::cout << rotated6_image.size() << std::endl;
				cv::namedWindow("Rotated Image", CV_WINDOW_AUTOSIZE);
				cv::imshow("Rotated Image", rotated6_image);
				counter++;
			}
		}
		catch (const std::exception& ex)
		{
			std::cout << "Error during rotating the original image: " << ex.what() << "\n\n";
		}
	
	cv::String saveFile7 = imagepath + "trainaugmented/" + folderName + "/" + std::to_string(counter) + ".jpg";
	cv::imwrite( saveFile7,rotated6_image);


	//rotate the flipped(2) image anticlockwise 90 
	cv::Mat rotated_flipped7_image;
	try
	{
		cv::rotate(flipped1_image, rotated_flipped7_image , cv::ROTATE_90_COUNTERCLOCKWISE);

		if (visualize_progress)
		{
			std::cout << rotated_flipped7_image.size() << std::endl;
			cv::namedWindow("Rotated Flipped Image", CV_WINDOW_AUTOSIZE);
			cv::imshow("Rotated Flipped Image", rotated_flipped7_image);
			counter++;
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Error during flipping the original image: " << ex.what() << "\n\n";
	}

	cv::String saveFile8 = imagepath + "trainaugmented/" + folderName + "/" + std::to_string(counter)+ ".jpg";
	cv::imwrite( saveFile8,rotated6_image);

  }

}


