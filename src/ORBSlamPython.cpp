#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <sophus/se3.hpp>
#include <ORB_SLAM3/KeyFrame.h>
#include <ORB_SLAM3/Converter.h>
#include <ORB_SLAM3/Tracking.h>
#include <ORB_SLAM3/MapPoint.h>
#include "ORBSlamPython.h"


static void* init_ar() {

    Py_Initialize();

    import_array();
    return NULL;
}

template <class Scalar>
cv::Mat SE3ToCvMat(const Sophus::SE3<Scalar>& se3) {
  // get 4x4 homogenous matrix
  const Eigen::Matrix<Scalar, 4, 4> m = se3.matrix();
  // convert to cv::Mat
  cv::Mat cvMat(m.rows(), m.cols(), CV_32F);
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      cvMat.at<float>(i, j) = m(i, j);
    }
  }
  return cvMat;
}

template <class Scalar, int Rows, int Cols>
cv::Mat eigenToCvMat(const Eigen::Matrix<Scalar, Rows, Cols>& eigenMat) {
    cv::Mat cvMat(eigenMat.rows(), eigenMat.cols(), CV_32F);
    for (int i = 0; i < eigenMat.rows(); ++i) {
        for (int j = 0; j < eigenMat.cols(); ++j) {
            cvMat.at<float>(i, j) = eigenMat(i, j);
        }
    }
    return cvMat;
}


BOOST_PYTHON_MODULE(orbslam3)
{
    init_ar();

    boost::python::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    boost::python::enum_<ORB_SLAM3::Tracking::eTrackingState>("TrackingState")
        .value("SYSTEM_NOT_READY", ORB_SLAM3::Tracking::eTrackingState::SYSTEM_NOT_READY)
        .value("NO_IMAGES_YET", ORB_SLAM3::Tracking::eTrackingState::NO_IMAGES_YET)
        .value("NOT_INITIALIZED", ORB_SLAM3::Tracking::eTrackingState::NOT_INITIALIZED)
        .value("OK", ORB_SLAM3::Tracking::eTrackingState::OK)
        .value("LOST", ORB_SLAM3::Tracking::eTrackingState::LOST);
    
    boost::python::enum_<ORB_SLAM3::System::eSensor>("Sensor")
        .value("MONOCULAR", ORB_SLAM3::System::eSensor::MONOCULAR)
        .value("STEREO", ORB_SLAM3::System::eSensor::STEREO)
        .value("RGBD", ORB_SLAM3::System::eSensor::RGBD);

    boost::python::class_<ORBSlamPython, boost::noncopyable>("System", boost::python::init<const char*, const char*, boost::python::optional<ORB_SLAM3::System::eSensor>>())
        .def(boost::python::init<std::string, std::string, boost::python::optional<ORB_SLAM3::System::eSensor>>())
        .def("initialize", &ORBSlamPython::initialize)
        .def("load_and_process_mono", &ORBSlamPython::loadAndProcessMono)
        .def("process_image_mono", &ORBSlamPython::processMono)
        .def("load_and_process_stereo", &ORBSlamPython::loadAndProcessStereo)
        .def("process_image_stereo", &ORBSlamPython::processStereo)
        .def("load_and_process_rgbd", &ORBSlamPython::loadAndProcessRGBD)
        .def("process_image_rgbd", &ORBSlamPython::processRGBD)
        .def("shutdown", &ORBSlamPython::shutdown)
        .def("is_running", &ORBSlamPython::isRunning)
        .def("reset", &ORBSlamPython::reset)
        .def("set_mode", &ORBSlamPython::setMode)
        .def("set_use_viewer", &ORBSlamPython::setUseViewer)
        .def("get_keyframe_points", &ORBSlamPython::getKeyframePoints)
        .def("get_trajectory_points", &ORBSlamPython::getTrajectoryPoints)
        .def("get_tracked_mappoints", &ORBSlamPython::getTrackedMappoints)
        .def("get_tracking_state", &ORBSlamPython::getTrackingState)
        .def("get_num_features", &ORBSlamPython::getNumFeatures)
        .def("get_num_matched_features", &ORBSlamPython::getNumMatches)
        .def("save_settings", &ORBSlamPython::saveSettings)
        .def("load_settings", &ORBSlamPython::loadSettings)
        .def("save_settings_file", &ORBSlamPython::saveSettingsFile)
        .staticmethod("save_settings_file")
        .def("load_settings_file", &ORBSlamPython::loadSettingsFile)
        .staticmethod("load_settings_file");
}

ORBSlamPython::ORBSlamPython(std::string vocabFile, std::string settingsFile, ORB_SLAM3::System::eSensor sensorMode)
    : vocabluaryFile(vocabFile),
    settingsFile(settingsFile),
    sensorMode(sensorMode),
    system(nullptr),
    bUseViewer(false),
    bUseRGB(true)
{
    
}

ORBSlamPython::ORBSlamPython(const char* vocabFile, const char* settingsFile, ORB_SLAM3::System::eSensor sensorMode)
    : vocabluaryFile(vocabFile),
    settingsFile(settingsFile),
    sensorMode(sensorMode),
    system(nullptr),
    bUseViewer(false),
    bUseRGB(true)
{

}

ORBSlamPython::~ORBSlamPython()
{
}

bool ORBSlamPython::initialize()
{
    system = std::make_shared<ORB_SLAM3::System>(vocabluaryFile, settingsFile, sensorMode, bUseViewer);
    return true;
}

bool ORBSlamPython::isRunning()
{
    return system != nullptr;
}

void ORBSlamPython::reset()
{
    if (system)
    {
        system->Reset();
    }
}

bool ORBSlamPython::loadAndProcessMono(std::string imageFile, double timestamp)
{
    if (!system)
    {
        return false;
    }
    cv::Mat im = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (bUseRGB)
    {
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    }
    return this->processMono(im, timestamp);
}

boost::python::list ORBSlamPython::processMono(cv::Mat image, double timestamp)
{
    if (!system)
    {
        return boost::python::list();
    }
    if (image.data)
    {
        auto pose = system->TrackMonocular(image, timestamp);
        const Eigen::Matrix<float, 4, 4> m = pose.matrix();
        boost::python::list t;

        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                t.append(m(i, j));
            }
        }
        return t;
    }
    else
    {
        return boost::python::list();
    }
}

bool ORBSlamPython::loadAndProcessStereo(std::string leftImageFile, std::string rightImageFile, double timestamp)
{
    if (!system)
    {
        return false;
    }
    cv::Mat leftImage = cv::imread(leftImageFile, cv::IMREAD_COLOR);
    cv::Mat rightImage = cv::imread(rightImageFile, cv::IMREAD_COLOR);
    if (bUseRGB) {
        cv::cvtColor(leftImage, leftImage, cv::COLOR_BGR2RGB);
        cv::cvtColor(rightImage, rightImage, cv::COLOR_BGR2RGB);
    }
    return this->processStereo(leftImage, rightImage, timestamp);
}

bool ORBSlamPython::processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp)
{
    if (!system)
    {
        return false;
    }
    if (leftImage.data && rightImage.data) {
        cv::Mat pose = SE3ToCvMat(system->TrackStereo(leftImage, rightImage, timestamp));
        return !pose.empty();
    }
    else
    {
        return false;
    }
}

bool ORBSlamPython::loadAndProcessRGBD(std::string imageFile, std::string depthImageFile, double timestamp)
{
    if (!system)
    {
        return false;
    }
    cv::Mat im = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (bUseRGB)
    {
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    }
    cv::Mat imDepth = cv::imread(depthImageFile, cv::IMREAD_UNCHANGED);
    return this->processRGBD(im, imDepth, timestamp);
}

bool ORBSlamPython::processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp)
{
    if (!system)
    {
        return false;
    }
    if (image.data && depthImage.data)
    {
        cv::Mat pose = SE3ToCvMat(system->TrackRGBD(image, depthImage, timestamp));
        return !pose.empty();
    }
    else
    {
        return false;
    }
}

void ORBSlamPython::shutdown()
{
    if (system)
    {
        system->Shutdown();
        system.reset();
    }
}

ORB_SLAM3::Tracking::eTrackingState ORBSlamPython::getTrackingState() const
{
    if (system)
    {
        return static_cast<ORB_SLAM3::Tracking::eTrackingState>(system->GetTrackingState());
    }
    return ORB_SLAM3::Tracking::eTrackingState::SYSTEM_NOT_READY;
}

unsigned int ORBSlamPython::getNumFeatures() const
{
    if (system)
    {
        return system->GetTracker()->mCurrentFrame.mvKeys.size();
    }
    return 0;
}

unsigned int ORBSlamPython::getNumMatches() const
{
    if (system)
    {
        // This code is based on the display code in FrameDrawer.cc, with a little extra safety logic to check the length of the vectors.
        ORB_SLAM3::Tracking* pTracker = system->GetTracker();
        unsigned int matches = 0;
        unsigned int num = pTracker->mCurrentFrame.mvKeys.size();
        if (pTracker->mCurrentFrame.mvpMapPoints.size() < num)
        {
            num = pTracker->mCurrentFrame.mvpMapPoints.size();
        }
        if (pTracker->mCurrentFrame.mvbOutlier.size() < num)
        {
            num = pTracker->mCurrentFrame.mvbOutlier.size();
        }
        for(unsigned int i = 0; i < num; ++i)
        {
            ORB_SLAM3::MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP && !pTracker->mCurrentFrame.mvbOutlier[i] && pMP->Observations() > 0)
            {
                ++matches;
            }
        }
        return matches;
    }
    return 0;
}

boost::python::list ORBSlamPython::getKeyframePoints() const
{
    if (!system)
    {
        return boost::python::list();
    }

    // This is copied from the ORB_SLAM3 System.SaveKeyFrameTrajectoryTUM function, with some changes to output a python tuple.
    vector<ORB_SLAM3::KeyFrame*> vpKFs = system->GetKeyFrames();
    std::sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    boost::python::list trajectory;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        ORB_SLAM3::KeyFrame* pKF = vpKFs[i];

        // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = eigenToCvMat(pKF->GetRotation()).t();
        cv::Mat t = eigenToCvMat(pKF->GetCameraCenter());
        trajectory.append(boost::python::make_tuple(
                              pKF->mTimeStamp,
                              R.at<float>(0,0),
                              R.at<float>(0,1),
                              R.at<float>(0,2),
                              t.at<float>(0),
                              R.at<float>(1,0),
                              R.at<float>(1,1),
                              R.at<float>(1,2),
                              t.at<float>(1),
                              R.at<float>(2,0),
                              R.at<float>(2,1),
                              R.at<float>(2,2),
                              t.at<float>(2)
                              ));
    }

    return trajectory;
}

boost::python::list ORBSlamPython::getTrackedMappoints() const
{
    if (!system)
    {
        return boost::python::list();
    }
    
    // This is copied from the ORB_SLAM3 System.SaveTrajectoryKITTI function, with some changes to output a python tuple.
    vector<ORB_SLAM3::MapPoint*> Mps = system->GetTrackedMapPoints();
    
    boost::python::list map_points;
    for(size_t i=0; i<Mps.size(); i++)    {
        cv::Mat wp = eigenToCvMat(Mps[i]->GetWorldPos());
        map_points.append(boost::python::make_tuple(
            wp.at<float>(0,0),
            wp.at<float>(1,0),
            wp.at<float>(2,0)                          
            ));
        }

        return map_points;
}

boost::python::list ORBSlamPython::getTrajectoryPoints() const
{
    if (!system)
    {
        return boost::python::list();
    }

    // This is copied from the ORB_SLAM3 System.SaveTrajectoryKITTI function, with some changes to output a python tuple.
    vector<ORB_SLAM3::KeyFrame*> vpKFs = system->GetKeyFrames();
    std::sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);

    boost::python::list trajectory;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        ORB_SLAM3::KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        Eigen::Vector3f t = Twc.translation();

        trajectory.append(boost::python::make_tuple(
            pKF->mTimeStamp,
            t(0),
            t(1),
            t(2),
            q.x(),
            q.y(),
            q.z(),
            q.w()
        ));
    }

    return trajectory;
}

void ORBSlamPython::setMode(ORB_SLAM3::System::eSensor mode)
{
    sensorMode = mode;
}

void ORBSlamPython::setUseViewer(bool useViewer)
{
    bUseViewer = useViewer;
}

void ORBSlamPython::setRGBMode(bool rgb)
{
    bUseRGB = rgb;
}

bool ORBSlamPython::saveSettings(boost::python::dict settings) const
{
    return ORBSlamPython::saveSettingsFile(settings, settingsFile);
}

boost::python::dict ORBSlamPython::loadSettings() const
{
    return ORBSlamPython::loadSettingsFile(settingsFile);
}

bool ORBSlamPython::saveSettingsFile(boost::python::dict settings, std::string settingsFilename)
{
    cv::FileStorage fs(settingsFilename.c_str(), cv::FileStorage::WRITE);
    
    boost::python::list keys = settings.keys();
    for (int index = 0; index < boost::python::len(keys); ++index)
    {
        boost::python::extract<std::string> extractedKey(keys[index]);
        if (!extractedKey.check())
        {
            continue;
        }
        std::string key = extractedKey;
        
        boost::python::extract<int> intValue(settings[key]);
        if (intValue.check())
        {
            fs << key << int(intValue);
            continue;
        }
        
        boost::python::extract<float> floatValue(settings[key]);
        if (floatValue.check())
        {
            fs << key << float(floatValue);
            continue;
        }
        
        boost::python::extract<std::string> stringValue(settings[key]);
        if (stringValue.check())
        {
            fs << key << std::string(stringValue);
            continue;
        }
    }
    
    return true;
}

// Helpers for reading cv::FileNode objects into python objects.
boost::python::list readSequence(cv::FileNode fn, int depth=10);
boost::python::dict readMap(cv::FileNode fn, int depth=10);

boost::python::dict ORBSlamPython::loadSettingsFile(std::string settingsFilename)
{
    cv::FileStorage fs(settingsFilename.c_str(), cv::FileStorage::READ);
    cv::FileNode root = fs.root();
    if (root.isMap()) 
    {
        return readMap(root);
    }
    else if (root.isSeq())
    {
        boost::python::dict settings;
        settings["root"] = readSequence(root);
        return settings;
    }
    return boost::python::dict();
}


// ----------- HELPER DEFINITIONS -----------
boost::python::dict readMap(cv::FileNode fn, int depth)
{
    boost::python::dict map;
    if (fn.isMap()) {
        cv::FileNodeIterator it = fn.begin(), itEnd = fn.end();
        for (; it != itEnd; ++it) {
            cv::FileNode item = *it;
            std::string key = item.name();
            
            if (item.isNone())
            {
                map[key] = boost::python::object();
            }
            else if (item.isInt())
            {
                map[key] = int(item);
            }
            else if (item.isString())
            {
                map[key] = std::string(item);
            }
            else if (item.isReal())
            {
                map[key] = float(item);
            }
            else if (item.isSeq() && depth > 0)
            {
                map[key] = readSequence(item, depth-1);
            }
            else if (item.isMap() && depth > 0)
            {
                map[key] = readMap(item, depth-1);  // Depth-limited recursive call to read inner maps
            }
        }
    }
    return map;
}

boost::python::list readSequence(cv::FileNode fn, int depth)
{
    boost::python::list sequence;
    if (fn.isSeq()) {
        cv::FileNodeIterator it = fn.begin(), itEnd = fn.end();
        for (; it != itEnd; ++it) {
            cv::FileNode item = *it;
            
            if (item.isNone())
            {
                sequence.append(boost::python::object());
            }
            else if (item.isInt())
            {
                sequence.append(int(item));
            }
            else if (item.isString())
            {
                sequence.append(std::string(item));
            }
            else if (item.isReal())
            {
                sequence.append(float(item));
            }
            else if (item.isSeq() && depth > 0)
            {
                sequence.append(readSequence(item, depth-1)); // Depth-limited recursive call to read nested sequences
            }
            else if (item.isMap() && depth > 0)
            {
                sequence.append(readMap(item, depth-1));
            }
        }
    }
    return sequence;
}
