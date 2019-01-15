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

// for NMS
#include "nms.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;


// global variables
int num_of_dtrees = 38;
float prob_threshold[] = {0.73, 0.77, 0.83};
float nms_thresh = 0.01;

int stride = 5;
std::vector<cv::Size> win_sizes = {cv::Size(80,80), cv::Size(110,110), cv::Size(130,130), cv::Size(150,150)};

int tot_classes = 4;
int num_train_files[] = {53, 81, 51, 290};
int min_images_in_a_class = 51;

int num_train_new_files[] = {424, 648, 408, 2320};
int min_images_in_a_class_new = 300;


// function declarations
int iou(std::vector<float> gt, cv::Rect r2, float threshold);
void precision_and_recall(float & precision, float & recall, 
                            std::vector<cv::Rect> & red_rect_0, 
                            std::vector<cv::Rect> & red_rect_1,
                            std::vector<cv::Rect> & red_rect_2,
                            std::string image_num);
void calc_hog(Mat image, vector<float>& descriptorsValues);
void get_data_and_labels(vector<Mat1f>& label_per_feats, Mat& labels);
void create_forest(Ptr<cv::ml::DTrees> * tree, int num_of_dtrees);
void train_forest(vector<Mat1f> label_per_feats, Ptr<cv::ml::DTrees> * tree, int num_of_dtrees);
void predict(Ptr<cv::ml::DTrees> * tree, int num_of_dtrees, Mat& test_features, 
             std::vector< std::vector<float> > & final_locations, 
             std::vector< std::vector<float> > & locations);

int get_final_locations(std::string file_path, std::vector< std::vector<float> > & final_locations,
                        Ptr<cv::ml::DTrees> * tree, int num_of_dtrees);


int main(int argc, char** argv){

    vector<Mat1f> label_per_feats(tot_classes);
    Mat labels;
    get_data_and_labels(label_per_feats, labels);
    
    Ptr<cv::ml::DTrees> tree[num_of_dtrees];

    create_forest(tree, num_of_dtrees);
    
    train_forest(label_per_feats, tree, num_of_dtrees);
    

    std::vector< std::vector<float> > final_locations;
//    std::string image_path = std::string("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/TDCV/TDCV/data/task3/test/") + std::string(argv[1]) +std::string(".jpg");
    std::string image_path = std::string("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/TDCV/TDCV/data/task3/test/")   + std::string("0000.jpg");
//    std::string file_path = std::string("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/TDCV/TDCV/data/task3/gt/") + std::string(argv[1]) +std::string(".gt.txt");
    std::string file_path = std::string("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/TDCV/TDCV/data/task3/gt/")  + std::string("0000.gt.txt");
    
    
    get_final_locations(image_path, final_locations, tree, num_of_dtrees);
    
    std::vector< std::vector<float> > rect_0, rect_1, rect_2;
    
    for (int i = 0; i < final_locations.size(); i++)
    {
        std::vector<float> temp = { final_locations.at(i).at(1), 
                                    final_locations.at(i).at(2), 
                                    final_locations.at(i).at(1) + final_locations.at(i).at(3), 
                                    final_locations.at(i).at(2) + final_locations.at(i).at(4)};

        switch (int(final_locations.at(i).at(0)))
        {
            case 0:
                rect_0.push_back(temp);
                break;
            case 1:
                rect_1.push_back(temp);
                break;
            case 2:
                rect_2.push_back(temp);
                break;
        }
    }
    std::cout<<"total rect num : "<< final_locations.size() << std::endl;
    std::cout<<"rect_0 size: "<< rect_0.size() << std::endl;
    std::cout<<"rect_1 size: "<< rect_1.size() << std::endl;
    std::cout<<"rect_2 size: "<< rect_2.size() << std::endl;

    std::vector<cv::Rect> reducedRectangle_0 = nms(rect_0, nms_thresh);
    std::vector<cv::Rect> reducedRectangle_1 = nms(rect_1, nms_thresh);
    std::vector<cv::Rect> reducedRectangle_2 = nms(rect_2, nms_thresh);

    cv::Mat image = cv::imread(image_path);
    
    cv::namedWindow( "label - 0", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "label - 1", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "label - 2", cv::WINDOW_AUTOSIZE );
    
    Mat image_clone_1, image_clone_2;
    image_clone_1 = image.clone();
    image_clone_2 = image.clone();

    DrawRectangles(image, reducedRectangle_0);
    DrawRectangles(image_clone_1, reducedRectangle_1);
    DrawRectangles(image_clone_2, reducedRectangle_2);
    
    cv::imshow("label - 0", image);
    cv::imshow("label - 1", image_clone_1);
    cv::imshow("label - 2", image_clone_2);
    
    float precision = 0, recall = 0;

    precision_and_recall(precision, recall, reducedRectangle_0, reducedRectangle_1, reducedRectangle_2, file_path);
    std::cout << "\n precision: " << precision << std::endl;
    std::cout << "\n recall: " << recall << std::endl;
//    cv::waitKey(0);
    return 0;
}

// intersection over union function
int iou(std::vector<float> gt, cv::Rect r2, float threshold)
{
    cv::Rect2d rect_1(gt[0], gt[1], gt[2] - gt[0], gt[3] - gt[1]);
    cv::Rect2d rect_2(r2.x, r2.y, r2.width, r2.height);

    std::cout << "rect_1: " << rect_1 << std::endl;
     std::cout << "rect_2: " << rect_2 << std::endl;

    cv::Rect2d r3 = rect_1 & rect_2;
    std::cout << "r3: " << r3 << "    " << r3.area() << std::endl;

    float overlap;
    if (r3.area() > 0)
    {
        overlap = r3.area()/ (rect_1.area()+rect_2.area()- r3.area());
        std::cout << "\noverlap:   " << overlap << std::endl;
    }
    else
        overlap = 0;

    if(overlap > threshold)
        return 1;
    else
        return 0;
}

// precision and recall calculation
void precision_and_recall(float & precision, float & recall, 
                            std::vector<cv::Rect> & red_rect_0, 
                            std::vector<cv::Rect> & red_rect_1,
                            std::vector<cv::Rect> & red_rect_2,
                            std::string image_num)
{
    std::vector< std::vector<float> > coordinates_gt;
    std::string line;
    std::fstream my_file(image_num.c_str(), std::ios::in);

    if (my_file.is_open())
    {
        int count = 0;
        while(count < 3)
        {
            getline(my_file, line);
            std::istringstream iss(line);
            std::vector<std::string> coordinates(std::istream_iterator<std::string>{iss},
                                                std::istream_iterator<std::string>());
            count++;
            coordinates_gt.push_back(std::vector<float>{std::stof(coordinates[1]), std::stof(coordinates[2]),
                                                        std::stof(coordinates[3]), std::stof(coordinates[4])});
        }
    }

    // calculating correct predictions for class 0
    int red_correct_0 = 0;
    int red_total_0 = 0;
    for (int i = 0; i < red_rect_0.size(); i++)
    {
        int iou_val = iou(coordinates_gt.at(0), red_rect_0.at(i), 0.5);
        if (iou_val == 1)
        {
            red_correct_0++;
        }
        red_total_0++;
        
    }

    // calculating correct predictions for class 1
    int red_correct_1 = 0;
    int red_total_1 = 0;
    for (int i = 0; i < red_rect_1.size(); i++)
    {
        int iou_val = iou(coordinates_gt.at(1), red_rect_1.at(i), 0.5);
        if (iou_val == 1)
        {
            red_correct_1++;
        }
        red_total_1++;
       
    }

    // calculating correct predictions for class 2
    int red_correct_2 = 0;
    int red_total_2 = 0;
    for (int i = 0; i < red_rect_2.size(); i++)
    {
        int iou_val = iou(coordinates_gt.at(2), red_rect_2.at(i), 0.5);
        if (iou_val == 1)
        {
            red_correct_2++;
        }
        red_total_2++;
        
    }

    // final precision and recall calc over an image
    precision = (red_correct_0 + red_correct_1 + red_correct_2) / (float)(red_total_0 + red_total_1 + red_total_2);
    recall = (red_correct_0 + red_correct_1 + red_correct_2) / (float)3;
}

// calculating hog descriptors for an image
void calc_hog(Mat image, vector<float>& descriptorsValues)
{
    
    try
    {
        cv::resize(image, image, Size(96, 96));
        cv::HOGDescriptor hog(Size(96, 96), Size(24, 24), Size(24, 24), Size(12, 12), 9);
        hog.compute(image, descriptorsValues, Size(0, 0), Size(0, 0));
    }
    catch (const std::exception& ex)
    {
        std::cout << "\nException during Reading image: " << ex.what() << std::endl;
    }
}

// creating feature and label data for training
void get_data_and_labels(vector<Mat1f>& label_per_feats, Mat& labels)
{
    String imagePath("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/TDCV/TDCV/data/task3/");
    std::vector<float> descriptorsValues;

    for (int i = 0; i < tot_classes; i++)
    {
        String folderName = "0" + std::to_string(i); 

        for (int j = 0; j < num_train_new_files[i]; j++)
        {
            String imageName;
            imageName = std::to_string(j) + ".jpg";
            String imgFile = imagePath + "train_new/" + folderName + "/" + imageName;
            //cout<<imgFile;
            cv::Mat image = cv::imread(imgFile, IMREAD_GRAYSCALE);
            //descriptorsValues = 
            calc_hog(image, descriptorsValues);
            Mat1f hog_descp(1, descriptorsValues.size(), descriptorsValues.data());
            label_per_feats[i].push_back(hog_descp);

            labels.push_back(i);        
        }
    }
}

// creating random forest
void create_forest(Ptr<cv::ml::DTrees> * tree, int num_of_dtrees)
{
    for (int idx = 0; idx < num_of_dtrees; idx++)
    {
        tree[idx] = cv::ml::DTrees::create();
    
        tree[idx]->setMaxDepth(20);
        tree[idx]->setMinSampleCount(5);
        tree[idx]->setCVFolds(0);
        tree[idx]->setMaxCategories(tot_classes);
    }
    
}

// training random forest
void train_forest(vector<Mat1f> label_per_feats, Ptr<cv::ml::DTrees> * tree, int num_of_dtrees)
{
    vector<Mat1f> feat_trainset_per_tree(num_of_dtrees);
    vector<Mat> labels_trainset_per_tree(num_of_dtrees);
    std::vector<int> indices;
    //cout<<"train forest initialisation passed \n";
    for (int curr_tree = 0; curr_tree < num_of_dtrees; ++curr_tree)
    {
        for (int curr_class = 0; curr_class < tot_classes; curr_class++)
        {
            for (int j = 0; j < num_train_new_files[curr_class]; j++) { 
                indices.push_back(j); 
            }
            cv::randShuffle(indices);
            
            for (int i = 0; i < min_images_in_a_class_new; ++i)
            {
                feat_trainset_per_tree[curr_tree].push_back(label_per_feats[curr_class].row(indices[i]));
                labels_trainset_per_tree[curr_tree].push_back(curr_class);
            }
            indices.clear();
        }
        
    }
    
    //cout<<"train forest first for loop passed \n";
    

    //TRAININIG HERE
    for (int idx = 0; idx < num_of_dtrees; idx++)
    {
        std::cout << "\n\ntraining tree " << idx <<std::endl;
        tree[idx]->train(cv::ml::TrainData::create(feat_trainset_per_tree[idx], cv::ml::ROW_SAMPLE, labels_trainset_per_tree[idx]));
    }
}

// random forest prediction and sends final locations after applying probability threshold
void predict(Ptr<cv::ml::DTrees> * tree, int num_of_dtrees, Mat& test_features, 
            std::vector< std::vector<float> > & final_locations, 
            std::vector< std::vector<float> > & locations)
{
    int test_class = 4;
    float correct, wrong;
    correct = wrong = 0;
    

    for (int j = 0; j < test_features.rows; j++)
    {
        float curr;
        Mat1f waste_array;

        int predictd_class[tot_classes];
        std::memset(predictd_class, 0, sizeof(predictd_class));
        
        
        for (int tree_idx = 0; tree_idx < num_of_dtrees; tree_idx++)
        {
            curr = tree[tree_idx]->predict(test_features.row(j), waste_array);
            predictd_class[(int)curr]++;
        }

        for (int i = 0;i< tot_classes - 1; ++i){
            if ( predictd_class[i]/(float)num_of_dtrees > prob_threshold[i]){
                std::vector<float> temp = {(float)i, locations.at(j).at(0), locations.at(j).at(1), locations.at(j).at(2), locations.at(j).at(3)};
                final_locations.push_back(temp);
            }
        }
    }
}

// reads an image and creates windows
int get_final_locations(std::string file_path, std::vector< std::vector<float> > & final_locations, Ptr<cv::ml::DTrees> * tree, int num_of_dtrees)
{
    cv::Mat image, crop;
	cv::Rect roi;
    
    image = cv::imread(file_path, IMREAD_GRAYSCALE);
    int count = 0, count_refined = 0;
    
    for (int win = 0; win < win_sizes.size(); win++)
    {
        cv::Mat test_data;
        std::vector< std::vector<float> > locations;

        cv::Size s = win_sizes[win];
        int col_length = (image.rows - s.height)/stride + 1;
        int row_length = (image.cols - s.width)/stride + 1;
        count = 0;

        for (int r = 0; r < row_length; r++)
        {
            for (int col = 0; col < col_length; col++)
            {
                roi = cv::Rect(r * stride, col * stride, s.width, s.height);

                image(roi).copyTo(crop);
                std::vector<float> descriptor;
                calc_hog(crop, descriptor);
                
                Mat1f row(1, descriptor.size(), descriptor.data());
                
                test_data.push_back(row);
                
                std::vector<float> temp = {(float)r * stride, (float)col * stride, (float)s.width, (float)s.height};
                locations.push_back(temp);
                
                descriptor.clear();
            }
        }
        predict(tree, num_of_dtrees, test_data, final_locations, locations);
        locations.clear();
    }
    return 0;
}
