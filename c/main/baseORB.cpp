#include <iostream>
#include <fstream>
#include <sstream>

#include "SLAMBase.hpp"

using namespace std;


int main (int argc, char** argv)
{
  // argv[1]: ymal
  // argv[2]: dataset path
  cv::FileStorage fSettings(argv[1], cv::FileStorage::READ);

  Camera_Intrinsic_Parameters camera;
  camera.scale = fSettings["DepthMapFactor"];
  camera.cx = fSettings["Camera.cx"];
  camera.cy = fSettings["Camera.cy"];
  camera.fx = fSettings["Camera.fx"];
  camera.fy = fSettings["Camera.fy"];

  int min_inliers = 5;
  double max_norm = 0.3;

  // bool visualize = true;
  bool visualize = false;

  vector<string> vstrImageFilenamesRGB;
  vector<string> vstrImageFilenamesD;
  vector<double> vTimestamps;

  string db_path = string(argv[2]);

  // pcl::visualization::CloudViewer viewer("viewer");

  LoadImages(db_path+"/associate.txt", vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

  int n_image = vTimestamps.size();


  ofstream outfile;
  outfile.open("base_evaluate.txt");
  outfile << "# benchmark evaluate use origial orb" << endl;
  outfile << "# timestamp tx ty tz qx qy qz qw" << endl;


  Frame lastFrame;
  PointCloud::Ptr cloud;

  cv::Mat Tcw;

  for (int i = 0; i < n_image; i++) {
    cout << "Index: " << i << endl;
    string rgb_path = db_path+"/"+vstrImageFilenamesRGB[i];
    string depth_path = db_path+"/"+vstrImageFilenamesD[i];

    Frame currFrame = Frame(i, rgb_path, depth_path);
    currFrame.ComputeFeatAndDesp();

    if (i == 0) {
      lastFrame = currFrame;
    //  cloud = Image2PointCloud(lastFrame.rgb, lastFrame.depth, camera);
      outfile << fixed << setprecision(6) << vTimestamps[i] << " " <<  setprecision(9) << 0  << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 1 << endl;
      continue;
    }
    // cv::waitKey(0);
    // 比较currFrame 和 lastFrame

    Result_of_PnP result = MatchAndRansac(lastFrame, currFrame, camera);
    if (result.inlPoint < min_inliers) {
      cout << "Warning: Number of inline points is too little." << endl;
      continue;
    } //inliers不够，放弃该帧
    // 计算运动范围是否太大

    double norm = normofTransform(result.rvec, result.tvec);
    cout<<"norm = "<<norm<<endl;
    if ( norm >= max_norm ) {
      cout << "Warning: Transform is too large." << endl;
      continue;
    }

    Eigen::Isometry3d T = RvecTvec2Mat(result.rvec, result.tvec);
    // cout<<"T = "<<T.matrix()<<endl;

    cv::Mat Tcr;
    cv::eigen2cv(T.matrix(), Tcr);
    if (i == 1) {
      Tcw = Tcr;
    }
    else {
      Tcw = Tcr*Tcw;
      cout << Tcw << endl;
    }
    cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
    cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

    vector<double> q = toQuaternion(Rwc);

    outfile << fixed << setprecision(6) << vTimestamps[i] << " " <<  setprecision(9) << twc.at<double>(0) << " " << twc.at<double>(1) << " " << twc.at<double>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;


    //cloud = joinPointCloud( cloud, currFrame, T.inverse(), camera );
    // if (visualize) {
    //   cloud = UpdatePointCloud( cloud, currFrame, T, camera );
    // }

    // if ( visualize && (i % 10 == 0) )
    //     viewer.showCloud( cloud );

    lastFrame = currFrame;
  }

  // string txt_file = argv[1];
  // int last_slash = txt_file.rfind('/') + 1;
  // string txt_base = txt_file.substr(0, last_slash);

  // ifstream infile;
  // infile.open(txt_file);

  // ofstream outfile;
  // outfile.open("base_evaluate.txt");
  // outfile << "# benchmark evaluate use origial orb" << endl;
  // outfile << "# timestamp tx ty tz qx qy qz qw" << endl;

  // cout << "Using File " << txt_file << endl;

  // string line;
  // while (getline(infile, line)){
  //   string str_tmp;
  //   string rgb_path;
  //   string depth_path;

  //   istringstream istr(line);

  //   istr >> str_tmp;
  //   double rgb_time = atof(str_tmp.c_str());

  //   istr >> rgb_path;

  //   istr >> str_tmp;
  //   double depth_time = atof(str_tmp.c_str());

  //   istr >> depth_path;

  //   outfile << fixed << (rgb_time + depth_time) / 2 << "  "+line << endl;
  //   // cout << rgb_time << "@" << txt_base+rgb_path << endl;
  //   // cout << depth_time << "@" << txt_base+depth_path << endl;
  //   // cout << line << endl;
  // }

  // infile.close();
}

