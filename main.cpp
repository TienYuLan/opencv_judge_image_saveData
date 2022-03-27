#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;
using namespace dnn;

//trainingDatas[image][image feat]
float trainingDatas[140][22];
int numImage = 140;
int imageData = 22;

const int POSE_PAIRS[3][20][2] = {
    {// COCO body
     {1, 2},
     {1, 5},
     {2, 3},
     {3, 4},
     {5, 6},
     {6, 7},
     {1, 8},
     {8, 9},
     {9, 10},
     {1, 11},
     {11, 12},
     {12, 13},
     {1, 0},
     {0, 14},
     {14, 16},
     {0, 15},
     {15, 17}},
    {// MPI body
     {0, 1},
     {1, 2},
     {2, 3},
     {3, 4},
     {1, 5},
     {5, 6},
     {6, 7},
     {1, 14},
     {14, 8},
     {8, 9},
     {9, 10},
     {14, 11},
     {11, 12},
     {12, 13}},
    {
        // hand
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4}, // thumb
        {0, 5},
        {5, 6},
        {6, 7},
        {7, 8}, // pinkie
        {0, 9},
        {9, 10},
        {10, 11},
        {11, 12}, // middle
        {0, 13},
        {13, 14},
        {14, 15},
        {15, 16}, // ring
        {0, 17},
        {17, 18},
        {18, 19},
        {19, 20} // small
    }};

string path = "C:/im/train/data";

double calDegree(Point pt1, Point pt2)
{
    double x, y, outcome, val;
    x = pt1.x - pt2.x;
    y = pt1.y - pt2.y;
    val = 180 / CV_PI;

    outcome = atan(y / x) * val;

    return outcome;
}

float *findData(string image)
{
    //image features
    static float data[22];
    //Calculate the number of line in the graph
    Mat src = imread(image);
    if (src.empty())
    {
        cerr << "Can't read image from the file: " << src << endl;
        exit(-1);
    }

    //resize(src, src, Size(256, 256));

    Mat edges;
    float vertical, horizontal, right, left;
    vertical = horizontal = right = left = 0.0;

    cvtColor(src, edges, CV_BGR2GRAY);
    GaussianBlur(edges, edges, Size(3, 3), 0, 0);
    Canny(edges, edges, 80, 300);
    vector<Vec2f> lines;
    int threshold = 80;
    HoughLines(edges, lines, 1, CV_PI / 180, threshold, 0, 0);

    for (size_t i = 0; i < lines.size(); i++)
    {
        double rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 500 * (-b));
        pt1.y = cvRound(y0 + 500 * (a));
        pt2.x = cvRound(x0 - 500 * (-b));
        pt2.y = cvRound(y0 - 500 * (a));

        if ((calDegree(pt1, pt2) >= 0 && calDegree(pt1, pt2) <= 15) || (calDegree(pt1, pt2) <= 0 && calDegree(pt1, pt2) >= -15))
        {
            horizontal += 1;
        }
        else if ((calDegree(pt1, pt2) >= 75 && calDegree(pt1, pt2) <= 90) || calDegree(pt1, pt2) <= -75)
        {
            vertical += 1;
        }
        else if (calDegree(pt1, pt2) > 15 && calDegree(pt1, pt2) < 75)
        {
            left += 1;
        }
        else
        {
            right += 1;
        }
    }

    if (lines.size() != 0)
    {
        data[0] = vertical / lines.size();
        data[1] = horizontal / lines.size();
        data[2] = right / lines.size();
        data[3] = left / lines.size();
    }
    else
    {
        data[0] = data[1] = data[2] = data[3] = 0;
    }

    //----------------------------//

    //----------------------------//
    // openpose
    String modelTxt = "D:/openpose-master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
    String modelBin = "D:/openpose-master/models/pose/mpi/pose_iter_160000.caffemodel";

    int W_in = 368;
    int H_in = 368;
    float thresh = (float)0.1;
    float scale = (float)0.003922;

    Mat img = src;

    if (modelTxt.empty() || modelBin.empty() || img.empty())
    {
        cout << "A sample app to demonstrate human or hand pose detection with a pretrained OpenPose dnn." << endl;
        return 0;
    }

    int npairs, nparts;
    npairs = 14;
    nparts = 16;

    // read the network model
    Net net = readNet(modelBin, modelTxt);
    // send it through the network
    Mat inputBlob = blobFromImage(img, scale, Size(W_in, H_in), Scalar(0, 0, 0), false, false);
    net.setInput(inputBlob);
    Mat result = net.forward();
    // the result is an array of "heatmaps", the probability of a body part being in location x,y
    int H = result.size[2];
    int W = result.size[3];

    // find the position of the body parts
    vector<Point> points(22);
    for (int n = 0; n < nparts; n++)
    {
        // Slice heatmap of corresponding body's part.
        Mat heatMap(H, W, CV_32F, result.ptr(0, n));
        // 1 maximum per heatmap
        Point p(-1, -1), pm;
        double conf;
        minMaxLoc(heatMap, 0, &conf, 0, &pm);
        if (conf > thresh)
            p = pm;
        p.x *= float(img.cols) / W;
        p.y *= float(img.rows) / H;
        points[n] = p;
    }
    // connect body parts and draw it !
    // float SX = float(img.cols) / W;
    // float SY = float(img.rows) / H;
    float lengthOfBodyParts[14];

    for (int n = 0; n < npairs; n++)
    {
        // lookup 2 connected body/hand parts
        Point2f partA = points[POSE_PAIRS[1][n][0]];
        Point2f partB = points[POSE_PAIRS[1][n][1]];

        // scale to image size
        // a.x *= SX;
        // a.y *= SY;
        // b.x *= SX;
        // b.y *= SY;

        Point2f c = partA - partB;

        // we did not find enough confidence before
        if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
        {
            lengthOfBodyParts[n] = 0;
        }
        else
        {
            lengthOfBodyParts[n] = (float)sqrt(pow(c.x, 2) + pow(c.y, 2));
        }
        //cout << lengthOfBodyParts[n] << endl;

        data[4 + n] = lengthOfBodyParts[n];
    }
    //----------------------------//
    //Get the proportion of the person's height, width, and center in the photo
    float fullBodyHeight, fullBodyWigth, fullBody_centerX, fullBody_centerY;
    fullBodyHeight = fullBodyWigth = fullBody_centerX = fullBody_centerY = 0.0;

    vector<string> classes;
    ifstream file("D:/darknet-master/data/coco.names");
    string line;
    while (getline(file, line))
    {
        classes.push_back(line);
    }

    Net net_yolo = readNetFromDarknet("D:/darknet-master/cfg/yolov4.cfg", "D:/darknet-master/build/darknet/x64/weights/yolov4.weights");

    DetectionModel model = DetectionModel(net_yolo);
    model.setInputParams(1 / 255.0, Size(416, 416), Scalar(), true);

    vector<int> classIds;
    vector<float> scores;
    vector<Rect> boxes;
    model.detect(src, classIds, scores, boxes, 0.6f, 0.4f);

    int max = 0;
    for (int i = 0; i < classIds.size(); i++)
    {
        int boxArea = boxes[i].height * boxes[i].width;
        if (classIds[i] == 0 && boxArea > max)
        {
            max = boxArea;
            fullBodyHeight = (float)boxes[i].height / (float)src.cols;
            fullBodyWigth = (float)boxes[i].width / (float)src.rows;
            fullBody_centerX = ((float)boxes[i].x + ((float)boxes[i].width / 2)) / (float)src.rows;
            fullBody_centerY = ((float)boxes[i].y + ((float)boxes[i].height / 2)) / (float)src.cols;
        }
    }
    data[18] = fullBodyHeight;
    data[19] = fullBodyWigth;
    data[20] = fullBody_centerX;
    data[21] = fullBody_centerY;

    //----------------------------//
    // cout << "data{" ;
    for (int i = 0; i < imageData; i++)
    {
        if (data[i] < 0.001)
        {
            data[i] = 0;
        }
    //cout << data[i] << ", ";
    }
    //cout << "}" << endl;
    return data;
}

void getImageData()
{
    //findData("C:/im/ann_train/data1.jpg");
    int i = 0;
    do
    {
        string num_cstr(to_string(i + 1));
        string s = path + num_cstr + ".jpg";
        float *temp;
        cout << "Loading photo : " << s << "........." << endl;
        temp = findData(s);
        cout << "Loading Completed" << endl;
        for (int j = 0; j < imageData; j++)
        {
            trainingDatas[i][j] = *(temp + j);
        }
        i++;
    } while (i < numImage);
}
//Mat -> csv
void saveMat(cv::Mat inputMat, char *filename)
{
    FILE *fpt = fopen(filename, "w");
    int rows = inputMat.rows;
    int clos = inputMat.cols;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < clos; j++)
        {
            if (j < clos - 1)
                fprintf(fpt, "%f,", inputMat.at<float>(i, j));
            else
                fprintf(fpt, "%f\n", inputMat.at<float>(i, j));
        }
    }
    fclose(fpt);
}

int main()
{
    cout << "get ImageData......" << endl;
    getImageData();
    cout << "Completed......" << endl;

    // make data[4] ~ data[17]'s value between 0 ~ 1
    for(int d = 4; d < 18; d++){
        // d = feature , i = numImage
        float max_data_value = 0;
        for (int i = 0; i < numImage; i++)
        {
            if (max_data_value < trainingDatas[i][d])
            {
                max_data_value = trainingDatas[i][d];
            }
        }
        for (int i = 0; i < numImage; i++)
        {
            trainingDatas[i][d] /= max_data_value;
        }
    }
    
    //float labels[100] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};

    
        float labels[140] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                             1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
                             1, 1, 1, 0, 0, 0, 1, 0, 1, 1,
                             1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
                             1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                             1, 0, 0, 1, 0, 1, 1, 0, 1, 1,
                             1, 0, 0, 1, 0, 1, 1, 1, 0, 0,
                             1, 1, 0, 1, 0, 1, 1, 1, 1, 0,
                             1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
                             1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
                             0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    Mat trainingDataMat(numImage, imageData, CV_32FC1, trainingDatas);
    Mat labelsMat(numImage, 1, CV_32FC1, labels);

    saveMat(trainingDataMat, "trainingDataMat.csv");
    saveMat(labelsMat, "labelsMat.csv");

}
