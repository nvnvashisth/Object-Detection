// This code calculate the hog descriptor, creates empty forest, train the forest and then predict the value. Correspondingly sanity is ensured by confidence and the Accuracy.
// NOTE: Change the path according to the input data


#include <iostream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/core.hpp"

// for HOGDescriptor class
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;
using namespace std;

// global declarations
int tot_classes=6;
int num_train_files[] = {49, 67, 42, 53, 67, 110};

// functions declarations
// 1. Hog descriptor calculation, 2. Read data and corresponding labels
// 3. Create forest (Empty forest creation), 4. Data load into the forest and training
// 5. Predict (Calculate the confidence and Accuracy)

void calc_hog_desc(Mat image, vector<float>& descriptorsValues);
void get_data_and_labels(vector<Mat1f>& label_per_feats, Mat& labels);
void create_forest(Ptr<cv::ml::DTrees> * tree, int num_of_dtrees);
void train_forest(vector<Mat1f> label_per_feats, Ptr<cv::ml::DTrees> * tree, int num_of_dtrees);
void predict(Ptr<cv::ml::DTrees> * tree, int num_of_dtrees);


int main()
{
    int num_of_dtrees = 64;
    vector<Mat1f> label_per_feats(tot_classes);
    Mat labels;
    
    // Function call
    get_data_and_labels(label_per_feats, labels);
    Ptr<cv::ml::DTrees> tree[num_of_dtrees];
    create_forest(tree, num_of_dtrees);
    train_forest(label_per_feats, tree, num_of_dtrees);
    predict(tree, num_of_dtrees);
    return 0;
}


void calc_hog_desc(Mat image, vector<float>& descriptorsValues)
{
//    Calculation of hog descriptor value and storing into the descriptorValue vector
    try
    {
        cv::resize(image, image, Size(96, 96));
        cv::HOGDescriptor hog(Size(96, 96), Size(24, 24), Size(24, 24), Size(12, 12), 9);
        hog.compute(image, descriptorsValues, Size(0, 0), Size(0, 0));
    }
    catch (const std::exception& ex)
    {
        std::cout << "\nException while reading image: " << ex.what() << std::endl;
    }
}

void get_data_and_labels(vector<Mat1f>& label_per_feats, Mat& labels){

//    Using the descriptorValues vector and appending with the label of different class.
    int tot_classes = 6;
    String imagePath("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/TDCV/TDCV/data/task2/");
    std::vector<float> descriptorsValues;

    for (int i = 0; i < tot_classes; i++)
    {
        String folderName = "0" + std::to_string(i);

        for (int j = 0; j < num_train_files[i]; j++)
        {
            String imageName;

            if (j < 10)
                imageName = "000" + std::to_string(j) + ".jpg";
            else if (j < 100)
                imageName = "00" + std::to_string(j) + ".jpg";
            else
                imageName = "0" + std::to_string(j) + ".jpg";

            String imgFile = imagePath + "train/" + folderName + "/" + imageName;
            cv::Mat image = cv::imread(imgFile, CV_LOAD_IMAGE_COLOR);
            
            calc_hog_desc(image, descriptorsValues);
            Mat1f hog_descp(1, descriptorsValues.size(), descriptorsValues.data());
            label_per_feats[i].push_back(hog_descp);

            labels.push_back(i);
        }
    }
}

void create_forest(Ptr<cv::ml::DTrees> * tree, int num_of_dtrees){
//    Creating a empty forest
    for (int idx = 0; idx < num_of_dtrees; idx++)
    {
        tree[idx] = cv::ml::DTrees::create();
    
        tree[idx]->setMaxDepth(20);
        tree[idx]->setMinSampleCount(5);
        tree[idx]->setCVFolds(0);
        tree[idx]->setMaxCategories(tot_classes);
    }
}

void train_forest(vector<Mat1f> label_per_feats, Ptr<cv::ml::DTrees> * tree, int num_of_dtrees)
{
//  Loading the shuffled data into the empty trees
    vector<Mat1f> feat_trainset_per_tree(num_of_dtrees);
    vector<Mat> labels_trainset_per_tree(num_of_dtrees);
    std::vector<int> indices;
    for (int curr_tree = 0; curr_tree < num_of_dtrees; ++curr_tree)
    {
        for (int curr_class = 0; curr_class < 6; curr_class++)
        {
            for (int j = 0; j < num_train_files[curr_class]; j++) { 
                indices.push_back(j); 
            }
            cv::randShuffle(indices);
            
            for (int i = 0; i < 42; i++)
            {
                feat_trainset_per_tree[curr_tree].push_back(label_per_feats[curr_class].row(indices[i]));
                labels_trainset_per_tree[curr_tree].push_back(curr_class);
            }
            indices.clear();
        }
    }

    //TRAININIG HERE
    for (int idx = 0; idx < num_of_dtrees; idx++)
    {
        std::cout << "\n\nTraining trees " << idx;
        tree[idx]->train(cv::ml::TrainData::create(feat_trainset_per_tree[idx], cv::ml::ROW_SAMPLE, labels_trainset_per_tree[idx]));
    }
}

void predict(Ptr<cv::ml::DTrees> * tree, int num_of_dtrees){

//    Class prediction and calculating the confidence and Accuracy.
    String imagePath("/Users/nitin/Documents/TUM/TUM/Sem_3/Tracking and Detection/Exercise/C++/TDCV/TDCV/data/task2/");
    int test_class = 6;
    int img_per_test_class = 10;
    String imageName;
    for (int i = 0; i < test_class; i++)
    {
        std::cout << "\n\npredicting class " << i << "\n";
        String folderName = "0" + std::to_string(i);
        float correct, wrong;
        correct = wrong = 0;
        float conf_aggr = 0;

        for (int j = 0; j < img_per_test_class; j++)
        {
            int curr_image = num_train_files[i] + j;

            if (curr_image < 100)
                imageName = "00" + std::to_string(num_train_files[i] + j) + ".jpg";
            else if (curr_image < 1000)
                imageName = "0" + std::to_string(num_train_files[i] + j) + ".jpg";

            String test_fileName = imagePath + "test/" + folderName + "/" + imageName;

            vector<float> test_descrip;
            cv::Mat testimage = cv::imread(test_fileName);
            calc_hog_desc(testimage, test_descrip);
            Mat1f hog_descp(1, test_descrip.size(), test_descrip.data());

            float curr;
            Mat1f waste_array;

            int predictd_class[tot_classes];
            std::memset(predictd_class, 0, sizeof(predictd_class));


            for (int tree_idx = 0; tree_idx < num_of_dtrees; tree_idx++)
            {
                curr = tree[tree_idx]->predict(hog_descp, waste_array);
                predictd_class[(int)curr]++;
            }

            int max_predicted_class = 0;

            for (int class_idx = 1; class_idx < tot_classes; class_idx++)
            {
                if (predictd_class[class_idx] > predictd_class[max_predicted_class])
                    max_predicted_class = class_idx;
  
            }

            float confidence = ((float)predictd_class[max_predicted_class] / (float)num_of_dtrees) * 100;
            
            conf_aggr = conf_aggr + confidence;
            
            if (max_predicted_class == i)
            {
                correct++;
            }

            else
                wrong++;

        }

        float accuracy = (float(correct) / float(correct + wrong));
        std::cout << "\n\nAccuracy: " << accuracy << std::endl;
        std::cout << "\n\nConfidence: " << conf_aggr/10 << std::endl;
        
    }
}
