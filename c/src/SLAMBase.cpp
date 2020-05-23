# include "SLAMBase.hpp"

PointCloud::Ptr Image2PointCloud( cv::Mat& rgb, cv::Mat& depth, Camera_Intrinsic_Parameters& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m+=2)
        for (int n=0; n < depth.cols; n+=2)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

cv::Point3f Point2dTo3d( cv::Point3f& point, Camera_Intrinsic_Parameters& camera )
{
    cv::Point3f p; // 3D 点
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}

Frame::Frame(void)
{
    ;
}

Frame::Frame(int index, string rgb_path, string depth_path)
{
    rgb = cv::imread(rgb_path);
    depth = cv::imread(depth_path, -1);
    frameID = index;
}

void Frame::ComputeFeatAndDesp(void)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);

    orb->detect(rgb, feat);
    orb->compute(rgb, feat, desp);

    return;
}

vector<cv::Mat> Frame::DescriptorVector(void)
{
    vector<cv::Mat> desps;
    for(size_t i = 0; i < feat.size(); i++)
    {
        desp.push_back(desp.row(i).clone());
    }
    return desps;
}

Result_of_PnP MatchAndRansac(Frame& frame1, Frame& frame2, Camera_Intrinsic_Parameters& camera)
{
	//read data from data floder
	cv::Mat pic1_rgb = frame1.rgb;
	cv::Mat pic2_rgb = frame2.rgb;
	cv::Mat pic1_depth = frame1.depth;
	cv::Mat pic2_depth = frame2.depth;
    cv::Mat pic1_desp = frame1.desp;
    cv::Mat pic2_desp = frame2.desp;

	vector< cv::KeyPoint > feat1 = frame1.feat;
	vector< cv::KeyPoint > feat2 = frame2.feat;

	//output the size of feature point
	std::cout<<"Key points of two images: "<<feat1.size()<<", "<<feat2.size()<<std::endl;

	//match all feature points
	vector< cv::DMatch > matches;
	cv::BFMatcher matcher;
	matcher.match(pic1_desp, pic2_desp, matches);
	std::cout<<"Find total "<<matches.size()<<" matches."<<std::endl;

	//output match
	cv::Mat imgMatch;
	// cv::drawMatches( pic1_rgb, feat1, pic2_rgb, feat2, matches, imgMatch);
	// cv::imshow( "matches", imgMatch);
	// cv::imwrite( "./data/matches.png", imgMatch);
	// cv::waitKey(0);

	//Selete feature point match
	//rule:delete points that the distance longer than forth min distance
    // first step : find the min distance
  vector< cv::DMatch > goodMatch;
  double min_dis = 10000000;

	for(size_t i = 0; i < matches.size(); i++)
	{
		if(matches[i].distance < min_dis) min_dis = matches[i].distance;
	}

	std::cout<<"min_dis = "<<min_dis<<std::endl;
  if(min_dis <= 20){
      min_dis = 20;
  }

	//second:slecte the good feature points
	for(size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance < 10*min_dis) goodMatch.push_back(matches[i]);
	}

	//output goodMatch
	std::cout<<"goodMatch = "<<goodMatch.size()<<std::endl;

	// cv::drawMatches( pic1_rgb, feat1, pic2_rgb, feat2, goodMatch, imgMatch);
	// cv::imshow( "good_matches", imgMatch);
	// cv::imwrite( "./data/good_matches.png", imgMatch);
	// cv::waitKey(0);

	//the next part: use the RANSAC to optimize

	//inital
	vector<cv::Point3f> pic_obj;
	vector<cv::Point2f> pic_img;

	//get the depth of pic1
	for(size_t i = 0; i < goodMatch.size(); i++)
	{
		cv::Point2f p = feat1[goodMatch[i].queryIdx].pt;
		ushort d = pic1_depth.ptr<ushort>(int(p.y))[int(p.x)];
		if(d == 0) continue;
		pic_img.push_back( cv::Point2f(feat2[goodMatch[i].trainIdx].pt));

		//(u,v,d) to (x,y,z)
		cv::Point3f pt (p.x, p.y, d);
		cv::Point3f pd = Point2dTo3d(pt, camera);
		pic_obj.push_back(pd);

	}

	double Camera_matrix[3][3] = {{camera.fx, 0, camera.cx},{0, camera.fy, camera.cy},{0, 0, 1}};

	//build the Camera matrix
	cv::Mat cameraMatrix(3, 3, CV_64F, Camera_matrix);
	cv::Mat rvec, tvec, inlPoint;

	cv::solvePnPRansac(pic_obj, pic_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 300, 8, 0.999, inlPoint);

	std::cout<<"inlPoint: "<<inlPoint.rows<<std::endl;
	std::cout<<"R="<<rvec<<std::endl;
	std::cout<<"t="<<tvec<<std::endl;

	vector<cv::DMatch> matchShow;
    for (size_t i = 0; i < inlPoint.rows; i++)
    {
        matchShow.push_back(goodMatch[inlPoint.ptr<int>(i)[0]]);
    }

	//out inlpoint
    // cv::drawMatches( pic1_rgb, feat1, pic2_rgb, feat2, matchShow, imgMatch);
    // cv::imshow( "inlPoint matches", imgMatch);
    // cv::imwrite( "./data/inlPoint.png", imgMatch);
    // cv::waitKey(0);

    Result_of_PnP result;
	result.rvec = rvec;
	result.tvec = tvec;
	result.inlPoint = inlPoint.rows;
	return result;
}

Result_of_PnP MatchAndRansac(Frame::Ptr& frame1, Frame::Ptr& frame2, Camera_Intrinsic_Parameters& camera)
{
    //read data from data floder
    cv::Mat pic1_rgb = frame1->rgb;
    cv::Mat pic2_rgb = frame2->rgb;
    cv::Mat pic1_depth = frame1->depth;
    cv::Mat pic2_depth = frame2->depth;
    cv::Mat pic1_desp = frame1->desp;
    cv::Mat pic2_desp = frame2->desp;

    vector< cv::KeyPoint > feat1 = frame1->feat;
    vector< cv::KeyPoint > feat2 = frame2->feat;

    //output the size of feature point
    std::cout<<"Key points of two images: "<<feat1.size()<<", "<<feat2.size()<<std::endl;

    //match all feature points
    vector< cv::DMatch > matches;
    cv::BFMatcher matcher;
    matcher.match(pic1_desp, pic2_desp, matches);
    std::cout<<"Find total "<<matches.size()<<" matches."<<std::endl;

    //output match
    cv::Mat imgMatch;
    // cv::drawMatches( pic1_rgb, feat1, pic2_rgb, feat2, matches, imgMatch);
    // cv::imshow( "matches", imgMatch);
    // cv::imwrite( "./data/matches.png", imgMatch);
    // cv::waitKey(0);

    //Selete feature point match
    //rule:delete points that the distance longer than forth min distance
    // first step : find the min distance
  vector< cv::DMatch > goodMatch;
  double min_dis = 10000000;

    for(size_t i = 0; i < matches.size(); i++)
    {
        if(matches[i].distance < min_dis) min_dis = matches[i].distance;
    }

    std::cout<<"min_dis = "<<min_dis<<std::endl;
  if(min_dis <= 20){
      min_dis = 20;
  }

    //second:slecte the good feature points
    for(size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < 10*min_dis) goodMatch.push_back(matches[i]);
    }

    //output goodMatch
    std::cout<<"goodMatch = "<<goodMatch.size()<<std::endl;

    // cv::drawMatches( pic1_rgb, feat1, pic2_rgb, feat2, goodMatch, imgMatch);
    // cv::imshow( "good_matches", imgMatch);
    // cv::imwrite( "./data/good_matches.png", imgMatch);
    // cv::waitKey(0);

    //the next part: use the RANSAC to optimize

    //inital
    vector<cv::Point3f> pic_obj;
    vector<cv::Point2f> pic_img;

    //get the depth of pic1
    for(size_t i = 0; i < goodMatch.size(); i++)
    {
        cv::Point2f p = feat1[goodMatch[i].queryIdx].pt;
        ushort d = pic1_depth.ptr<ushort>(int(p.y))[int(p.x)];
        if(d == 0) continue;
        pic_img.push_back( cv::Point2f(feat2[goodMatch[i].trainIdx].pt));

        //(u,v,d) to (x,y,z)
        cv::Point3f pt (p.x, p.y, d);
        cv::Point3f pd = Point2dTo3d(pt, camera);
        pic_obj.push_back(pd);

    }

    double Camera_matrix[3][3] = {{camera.fx, 0, camera.cx},{0, camera.fy, camera.cy},{0, 0, 1}};

    //build the Camera matrix
    cv::Mat cameraMatrix(3, 3, CV_64F, Camera_matrix);
    cv::Mat rvec, tvec, inlPoint;

    cv::solvePnPRansac(pic_obj, pic_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 300, 8, 0.999, inlPoint);

    std::cout<<"inlPoint: "<<inlPoint.rows<<std::endl;
    std::cout<<"R="<<rvec<<std::endl;
    std::cout<<"t="<<tvec<<std::endl;

    vector<cv::DMatch> matchShow;
    for (size_t i = 0; i < inlPoint.rows; i++)
    {
        matchShow.push_back(goodMatch[inlPoint.ptr<int>(i)[0]]);
    }

    //out inlpoint
    // cv::drawMatches( pic1_rgb, feat1, pic2_rgb, feat2, matchShow, imgMatch);
    // cv::imshow( "inlPoint matches", imgMatch);
    // cv::imwrite( "./data/inlPoint.png", imgMatch);
    // cv::waitKey(0);

    Result_of_PnP result;
    result.rvec = rvec;
    result.tvec = tvec;
    result.inlPoint = inlPoint.rows;
    return result;
}


Eigen::Isometry3d RvecTvec2Mat(cv::Mat& rvec, cv::Mat& tvec)
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0);
    T(1,3) = tvec.at<double>(1,0);
    T(2,3) = tvec.at<double>(2,0);
    return T;
}


PointCloud::Ptr UpdatePointCloud(PointCloud::Ptr last_pc, Frame& new_frame, Eigen::Isometry3d T, Camera_Intrinsic_Parameters& camera)
{

	PointCloud::Ptr newCloud = Image2PointCloud(new_frame.rgb, new_frame.depth, camera);
	PointCloud::Ptr output (new PointCloud());
	pcl::transformPointCloud( *last_pc, *output, T.matrix() );
	*newCloud += *output;
	std::cout<<"Done!!!"<<std::endl;

	static pcl::VoxelGrid<PointT> voxel;
	double gridsize = 0.01;
	voxel.setLeafSize( gridsize, gridsize, gridsize );
	voxel.setInputCloud(newCloud);
	PointCloud::Ptr tmp(new PointCloud());
	return newCloud;

}


double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<double>(0,0), cvMat3.at<double>(0,1), cvMat3.at<double>(0,2),
         cvMat3.at<double>(1,0), cvMat3.at<double>(1,1), cvMat3.at<double>(1,2),
         cvMat3.at<double>(2,0), cvMat3.at<double>(2,1), cvMat3.at<double>(2,2);

    return M;
}


std::vector<double> toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<double> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

