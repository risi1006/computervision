#include <raspicam/raspicam_cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/face.hpp>
#include <iostream>
 
using namespace std;
 
void detectFace(cv::Mat &frame,
        cv::CascadeClassifier &face_cascade,
        vector<cv::Rect> &faces,
        cv::Ptr<cv::face::FaceRecognizer> &model,
        int &pos_x,
        int &pos_y,
        string &text) {
  cv::Mat grayscale;
 
  // Konvertieren des Bildes: Graustufen, Normalisierung der Helligkeit und Erhöhung des Kontrastes
  cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(grayscale, grayscale);
 
  face_cascade.detectMultiScale(grayscale, faces, 1.6, 3, 0|CV_HAAR_SCALE_IMAGE, cv::Size(80, 80), cv::Size(200, 200));
 
  for (size_t i = 0; i < faces.size(); i++) { cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 ); cv::ellipse( frame, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 255, 255 ), 2, 8, 0 ); 


cv::Mat face_i = cv::Mat(frame, faces[i]); 
 
cv::cvtColor(face_i, face_i, cv::COLOR_BGR2GRAY); 
cv::resize(face_i, face_i, cv::Size(100, 100), 0, 0, CV_INTER_NN); 

double predicted_confidence = 0.0; int prediction = -1; 
model->predict(face_i,prediction,predicted_confidence);
    cout << "Prediction " << prediction << " Confidence " << predicted_confidence << endl;
    if (prediction == 0 && predicted_confidence < 110.0) {
      text = "Johannes";
      pos_x = max(faces[i].tl().x - 10, 0);
      pos_y = max(faces[i].tl().y - 10, 0);
      cv::putText( frame, text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
    } else if (prediction == 1 && predicted_confidence < 110.0) {
      text = "Alex";
      pos_x = max(faces[i].tl().x - 10, 0);
      pos_y = max(faces[i].tl().y - 10, 0);
      cv::putText( frame, text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
    } else if (prediction == 2 && predicted_confidence < 110.0) {
      text = "Christian";
      pos_x = max(faces[i].tl().x - 10, 0);
      pos_y = max(faces[i].tl().y - 10, 0);
      cv::putText( frame, text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
    } else {
      text = "Face not recognized... Sorry :(";
    }
  }
}
 
 
int main(int argc, char **argv) {

  // Kamera
  raspicam::RaspiCam_Cv Camera;

  // Bild
  cv::Mat frame;

  // Classifier
  cv::CascadeClassifier face_cascade;
  cv::Ptr<cv::face::FaceRecognizer> model = cv::face::createLBPHFaceRecognizer();

  // Vektor mit Rechtecken für jedes im Bild erkannte Gesicht
  vector<cv::Rect> faces;

  // Text der in der GUI dargestellt werden soll
  string text = "";

  // X-Position des erkannten Gesichtes
  int pos_x = 0;

  // Y-Position des erkannten Gesichtes
  int pos_y = 0;


  size_t i = 0;
 
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " <Detection File> <Recognition File>" << endl;
    return -1;
  }
 
  string classifier_file = argv[1];
  string face_model = argv[2];
 
  cout << "Loading face cascade.." << endl;
  if (!face_cascade.load(classifier_file)) {
    cerr << "Error loading face cascade!" << endl;
    return -1;
  }
 
  cout << "Loading face recognition model.." << endl; model->load(face_model);
   
  Camera.set(CV_CAP_PROP_FORMAT, CV_8UC3);
  Camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  Camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
 
  cout << "Opening camera..." << endl;
  if (!Camera.open()) {
    cerr << "Error opening camera!" << endl;
    return -1;
  }
   
  cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);
 
  for (;;i++) {
    Camera.grab();
    Camera.retrieve(frame);
 
    if (i % 6 == 0) {
      detectFace(frame,
         face_cascade,
         faces,
         model,
         pos_x,
         pos_y,
         text);
    } 

    else {
      for (size_t i = 0; i < faces.size(); i++) { 
		cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
		cv::ellipse( frame, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 255, 255 ), 2, 8, 0 ); 
		cv::putText( frame, text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	  }
    } 

  cv::imshow("Display Window", frame); if (cv::waitKey(1) > 0) {
	break;
    }
  }
   
  cout << "Stopping camera.." << endl;
  Camera.release();
  return 0;
}

//Camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
//Camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
