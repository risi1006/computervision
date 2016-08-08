#include <raspicam/raspicam_cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/face.hpp>
#include <iostream>
 
using namespace std;
 
void detectFace(cv::Mat &colorFrame,
        cv::CascadeClassifier &classifier,
        vector<cv::Rect> &facesRects,
        cv::Ptr<cv::face::FaceRecognizer> &lbphModel,
        int &facePositionX,
        int &facePositionY,
        string &displayText) {

  cv::Mat grayscaleFrame;
 
  // Konvertieren des Bildes: Graustufen, Normalisierung der Helligkeit und Erhöhung des Kontrastes
  cv::cvtColor(colorFrame, grayscaleFrame, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(grayscaleFrame, grayscaleFrame);

  // Alle Gesichter im Frame mittels Classifier erkennen und in Vector facesRects ablegen
  classifier.detectMultiScale(grayscaleFrame, facesRects, 1.6, 3, 0|CV_HAAR_SCALE_IMAGE, cv::Size(80, 80), cv::Size(200, 200));

  // Alle erkannten Gesichter werden mit einer Ellipse markiert
  for (size_t i = 0; i < facesRects.size(); i++) {
    cv::Point faceMid( facesRects[i].x + facesRects[i].width*0.5, facesRects[i].y + facesRects[i].height*0.5 );
    cv::ellipse( colorFrame, faceMid, cv::Size( facesRects[i].width*0.5, facesRects[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 255, 255 ), 2, 8, 0 );

    // Erkanntes Gesicht einzeln holen und vergroessern
    cv::Mat detectedFace = cv::Mat(colorFrame, facesRects[i]);
    cv::cvtColor(detectedFace, detectedFace, cv::COLOR_BGR2GRAY);
    cv::resize(detectedFace, detectedFace, cv::Size(100, 100), 0, 0, CV_INTER_NN);

    // Genauigkeit der Erkennung
    double confidence = 0.0;

    // Erkanntes Gesicht gemäß Trainings-Labels (0 = Johannes, 1 = Alex, 2 = Christian, 3 = unbekannt)
    int prediction = -1;

    // Gesicht identifizieren
      lbphModel->predict(detectedFace, prediction, confidence);

    if (prediction == 0 && confidence < 110.0) {
      displayText = "Johannes";
    }

    else if (prediction == 1 && confidence < 110.0) {
      displayText = "Alex";
    }

    else if (prediction == 2 && confidence < 110.0) {
      displayText = "Christian";
    }

    else {
      displayText = "Unbekannt";
    }

    cout << "Erkanntes Gesicht: " << displayText << " Genauigkeit: " << confidence << endl;

    // Text an Gesichtsposition im Frame setzen
    facePositionX = max(facesRects[i].tl().x - 10, 0);
    facePositionY = max(facesRects[i].tl().y - 10, 0);
    cv::putText( colorFrame, displayText, cv::Point(facePositionX, facePositionY), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
  }
}
 
 
int main(int argc, char **argv) {

  // Kamera-Objekt
  raspicam::RaspiCam_Cv camera;

  // Ausgabe-Frame
  cv::Mat colorFrame;

  // Classifier-Objekt
  cv::CascadeClassifier classifier;

  // LBPH-Modell
  cv::Ptr<cv::face::FaceRecognizer> lbphModel = cv::face::createLBPHFaceRecognizer();

  // Vektor mit Rechtecken für jedes im Bild erkannte Gesicht
  vector<cv::Rect> facesRects;

  // Text der in der GUI dargestellt werden soll
  string displayText = "";

  // X-Position des erkannten Gesichtes
  int facePositionX = 0;

  // Y-Position des erkannten Gesichtes
  int facePositionY = 0;

  size_t i = 0; // Schleifenzaehler


    // Prüfung auf richtige Anzahl der übergebenen Argumente
  if (argc != 3) {
    cerr << "Ungueltige oder fehlende Argumente..." << endl;
    return -1;
  }

  // Übergebene Argumente (1 = Classifier, 2 = LBPH-Modell)
  string fileClassifier = argv[1];
  string fileLbphModel = argv[2];

  if (!classifier.load(fileClassifier)) {
    cerr << "Classifier konnte nicht geladen werden..." << endl;
    return -1;
  }

  // Laden des Modells
  lbphModel->load(fileLbphModel);

  // Kamera-Einstellungen festlegen
  camera.set(CV_CAP_PROP_FORMAT, CV_8UC3); // Farb-Bild
  camera.set(CV_CAP_PROP_FRAME_WIDTH, 640); // Reduzierung des Kamera-Bildes auf 640x480 zur Performance-Optimierung
  camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  if (!camera.open()) {
    cerr << "Kamera konnte nicht gestartet werden..." << endl;
    return -1;
  }

  // Fenster erzeugen
  cv::namedWindow("ComputerVisionCam", cv::WINDOW_AUTOSIZE);
 
  for (;;i++) {
    camera.grab();
    camera.retrieve(colorFrame); // Kamera-Bilddaten holen
 
    if (i % 6 == 0) {

      // Aufruf der Gesichtserkennung
      detectFace(colorFrame,
         classifier,
         facesRects,
         lbphModel,
         facePositionX,
         facePositionY,
         displayText);
    } 

    else {
      for (size_t i = 0; i < facesRects.size(); i++) {
		cv::Point faceMid( facesRects[i].x + facesRects[i].width*0.5, facesRects[i].y + facesRects[i].height*0.5 );
		cv::ellipse( colorFrame, faceMid, cv::Size( facesRects[i].width*0.5, facesRects[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 255, 255 ), 2, 8, 0 );
		cv::putText( colorFrame, displayText, cv::Point(facePositionX, facePositionY), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	  }
    } 

  // Kamera-Bild im Fenster aktualisieren
  cv::imshow("ComputerVisionCam", colorFrame);
    if (cv::waitKey(1) > 0) {
	  break;
    }
  }
   
  cout << "Kamera-Aufnahme wird beendet..." << endl;
  camera.release();
  return 0;
}
