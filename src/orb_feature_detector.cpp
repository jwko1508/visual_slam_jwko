#include "orb_feature_detector.hpp"

#include <algorithm>
#include <chrono>
#include <list>
#include <string>

#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include "std_msgs/msg/header.hpp"

// ---------------------------------------------------------------------------
// ORB-SLAM3 스타일 쿼드트리 균등화 (DistributeOctTree)
// ---------------------------------------------------------------------------
namespace
{
    // -------------------------------------------------------------------------
    // ExtractorNode
    // -------------------------------------------------------------------------
    struct ExtractorNode
    {
        cv::Point2i UL, UR, BL, BR;
        std::vector<cv::KeyPoint> vKeys;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore{false};

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2,
                        ExtractorNode &n3, ExtractorNode &n4)
        {
            const int halfX = static_cast<int>(ceil(static_cast<float>(UR.x - UL.x) / 2.0f));
            const int halfY = static_cast<int>(ceil(static_cast<float>(BR.y - UL.y) / 2.0f));

            n1.UL = UL;
            n1.UR = cv::Point2i(UL.x + halfX, UL.y);
            n1.BL = cv::Point2i(UL.x, UL.y + halfY);
            n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);

            n2.UL = n1.UR;
            n2.UR = UR;
            n2.BL = n1.BR;
            n2.BR = cv::Point2i(UR.x, UL.y + halfY);

            n3.UL = n1.BL;
            n3.UR = n1.BR;
            n3.BL = BL;
            n3.BR = cv::Point2i(n1.BR.x, BR.y);

            n4.UL = n3.UR;
            n4.UR = n2.BR;
            n4.BL = n3.BR;
            n4.BR = BR;

            for (const auto &kp : vKeys)
            {
                if (kp.pt.x < n1.UR.x)
                {
                    if (kp.pt.y < n1.BL.y)
                        n1.vKeys.push_back(kp);
                    else
                        n3.vKeys.push_back(kp);
                }
                else
                {
                    if (kp.pt.y < n1.BL.y)
                        n2.vKeys.push_back(kp);
                    else
                        n4.vKeys.push_back(kp);
                }
            }
        }
    };

    bool compareNodes(const std::pair<int, ExtractorNode *> &e1,
                      const std::pair<int, ExtractorNode *> &e2)
    {
        return e1.first < e2.first;
    }

    // -------------------------------------------------------------------------
    // DistributeOctTree  (ORB-SLAM3 원본 로직, standalone 버전)
    // -------------------------------------------------------------------------
    std::vector<cv::KeyPoint> DistributeOctTree(
        const std::vector<cv::KeyPoint> &vToDistributeKeys,
        const int &minX, const int &maxX,
        const int &minY, const int &maxY,
        const int &N, const int & /*level*/)
    {
        const int nIni = std::max(1,
                                  static_cast<int>(round(static_cast<float>(maxX - minX) / (maxY - minY))));
        const float hX = static_cast<float>(maxX - minX) / nIni;

        std::list<ExtractorNode> lNodes;
        std::vector<ExtractorNode *> vpIniNodes(nIni);

        for (int i = 0; i < nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(static_cast<int>(hX * i), 0);
            ni.UR = cv::Point2i(static_cast<int>(hX * (i + 1)), 0);
            ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
            ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
            ni.vKeys.reserve(vToDistributeKeys.size());
            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        for (const auto &kp : vToDistributeKeys)
        {
            int idx = std::min(static_cast<int>(kp.pt.x / hX), nIni - 1);
            vpIniNodes[idx]->vKeys.push_back(kp);
        }

        auto lit = lNodes.begin();
        while (lit != lNodes.end())
        {
            if (lit->vKeys.size() == 1)
            {
                lit->bNoMore = true;
                ++lit;
            }
            else if (lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                ++lit;
        }

        bool bFinish = false;
        std::vector<std::pair<int, ExtractorNode *>> vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size() * 4);

        while (!bFinish)
        {
            int prevSize = static_cast<int>(lNodes.size());
            lit = lNodes.begin();
            int nToExpand = 0;
            vSizeAndPointerToNode.clear();

            while (lit != lNodes.end())
            {
                if (lit->bNoMore)
                {
                    ++lit;
                    continue;
                }

                ExtractorNode n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);

                auto push = [&](ExtractorNode &n)
                {
                    if (n.vKeys.empty())
                        return;
                    lNodes.push_front(n);
                    lNodes.front().lit = lNodes.begin();
                    if (static_cast<int>(n.vKeys.size()) > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(
                            {static_cast<int>(n.vKeys.size()), &lNodes.front()});
                    }
                };
                push(n1);
                push(n2);
                push(n3);
                push(n4);
                lit = lNodes.erase(lit);
            }

            if (static_cast<int>(lNodes.size()) >= N ||
                static_cast<int>(lNodes.size()) == prevSize)
            {
                bFinish = true;
            }
            else if (static_cast<int>(lNodes.size()) + nToExpand * 3 > N)
            {
                while (!bFinish)
                {
                    prevSize = static_cast<int>(lNodes.size());
                    auto vPrev = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();

                    std::sort(vPrev.begin(), vPrev.end(), compareNodes);

                    for (int j = static_cast<int>(vPrev.size()) - 1; j >= 0; --j)
                    {
                        ExtractorNode n1, n2, n3, n4;
                        vPrev[j].second->DivideNode(n1, n2, n3, n4);

                        auto push2 = [&](ExtractorNode &n)
                        {
                            if (n.vKeys.empty())
                                return;
                            lNodes.push_front(n);
                            lNodes.front().lit = lNodes.begin();
                            if (static_cast<int>(n.vKeys.size()) > 1)
                                vSizeAndPointerToNode.push_back(
                                    {static_cast<int>(n.vKeys.size()), &lNodes.front()});
                        };
                        push2(n1);
                        push2(n2);
                        push2(n3);
                        push2(n4);
                        lNodes.erase(vPrev[j].second->lit);

                        if (static_cast<int>(lNodes.size()) >= N)
                            break;
                    }

                    if (static_cast<int>(lNodes.size()) >= N ||
                        static_cast<int>(lNodes.size()) == prevSize)
                        bFinish = true;
                }
            }
        }

        // 각 리프 노드에서 response 최대 키포인트 1개 선택
        std::vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(N);
        for (auto &node : lNodes)
        {
            const auto best = std::max_element(
                node.vKeys.begin(), node.vKeys.end(),
                [](const cv::KeyPoint &a, const cv::KeyPoint &b)
                { return a.response < b.response; });
            if (best != node.vKeys.end())
                vResultKeys.push_back(*best);
        }
        return vResultKeys;
    }

    // -------------------------------------------------------------------------
    // ComputeKeyPointsOctTree  (ORB-SLAM3 멤버 함수 → standalone 포팅)
    // -------------------------------------------------------------------------
    std::vector<cv::KeyPoint> ComputeKeyPointsOctTree(
        const cv::Mat &image,
        int nfeatures, float scaleFactor, int nlevels,
        int edgeThreshold, int iniThFAST, int minThFAST)
    {
        // 피라미드 레벨별 목표 피처 수 (ORB-SLAM3 동일 공식)
        std::vector<int> mnFeaturesPerLevel(nlevels);
        {
            float factor = 1.0f / scaleFactor;
            float nDesired = static_cast<float>(nfeatures) * (1.0f - factor) /
                             (1.0f - std::pow(factor, static_cast<float>(nlevels)));
            int sum = 0;
            for (int level = 0; level < nlevels - 1; ++level)
            {
                mnFeaturesPerLevel[level] = cvRound(nDesired);
                sum += mnFeaturesPerLevel[level];
                nDesired *= factor;
            }
            mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sum, 0);
        }

        // 이미지 피라미드 구축
        std::vector<cv::Mat> mvImagePyramid(nlevels);
        mvImagePyramid[0] = image;
        for (int level = 1; level < nlevels; ++level)
        {
            float invScale = 1.0f / std::pow(scaleFactor, static_cast<float>(level));
            cv::resize(image, mvImagePyramid[level],
                       cv::Size(cvRound(image.cols * invScale),
                                cvRound(image.rows * invScale)),
                       0, 0, cv::INTER_LINEAR);
        }

        std::vector<cv::KeyPoint> allKeypoints;
        allKeypoints.reserve(nfeatures * 2);
        const float W = 35.0f;

        for (int level = 0; level < nlevels; ++level)
        {
            const int minBorderX = edgeThreshold - 3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols - edgeThreshold + 3;
            const int maxBorderY = mvImagePyramid[level].rows - edgeThreshold + 3;

            if (maxBorderX <= minBorderX || maxBorderY <= minBorderY)
                continue;

            std::vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures * 10);

            const float width = static_cast<float>(maxBorderX - minBorderX);
            const float height = static_cast<float>(maxBorderY - minBorderY);
            const int nCols = std::max(1, static_cast<int>(width / W));
            const int nRows = std::max(1, static_cast<int>(height / W));
            const int wCell = static_cast<int>(std::ceil(width / nCols));
            const int hCell = static_cast<int>(std::ceil(height / nRows));

            for (int i = 0; i < nRows; ++i)
            {
                const float iniY = minBorderY + i * hCell;
                float maxY = iniY + hCell + 6;
                if (iniY >= maxBorderY - 3)
                    continue;
                if (maxY > maxBorderY)
                    maxY = maxBorderY;

                for (int j = 0; j < nCols; ++j)
                {
                    const float iniX = minBorderX + j * wCell;
                    float maxX = iniX + wCell + 6;
                    if (iniX >= maxBorderX - 6)
                        continue;
                    if (maxX > maxBorderX)
                        maxX = maxBorderX;

                    std::vector<cv::KeyPoint> vKeysCell;
                    cv::FAST(
                        mvImagePyramid[level]
                            .rowRange(static_cast<int>(iniY), static_cast<int>(maxY))
                            .colRange(static_cast<int>(iniX), static_cast<int>(maxX)),
                        vKeysCell, iniThFAST, true);

                    if (vKeysCell.empty())
                        cv::FAST(
                            mvImagePyramid[level]
                                .rowRange(static_cast<int>(iniY), static_cast<int>(maxY))
                                .colRange(static_cast<int>(iniX), static_cast<int>(maxX)),
                            vKeysCell, minThFAST, true);

                    for (auto &kp : vKeysCell)
                    {
                        kp.pt.x += j * wCell;
                        kp.pt.y += i * hCell;
                        vToDistributeKeys.push_back(kp);
                    }
                }
            }

            // OctTree 균등 분포 → mnFeaturesPerLevel[level] 개
            std::vector<cv::KeyPoint> levelKps = DistributeOctTree(
                vToDistributeKeys,
                minBorderX, maxBorderX, minBorderY, maxBorderY,
                mnFeaturesPerLevel[level], level);

            // level-0 좌표로 변환 후 octave / size 설정
            const float sf = std::pow(scaleFactor, static_cast<float>(level));
            const int scaledPatchSize = static_cast<int>(31 * sf);
            for (auto &kp : levelKps)
            {
                kp.pt.x = (kp.pt.x + minBorderX) * sf;
                kp.pt.y = (kp.pt.y + minBorderY) * sf;
                kp.octave = level;
                kp.size = static_cast<float>(scaledPatchSize);
            }
            allKeypoints.insert(allKeypoints.end(), levelKps.begin(), levelKps.end());
        }

        return allKeypoints;
    }

} // anonymous namespace

namespace jwko
{
    namespace orb_feature_detector
    {

        OrbFeatureDetector::OrbFeatureDetector(const std::string &instance_name)
            : instance_name_(instance_name),
              node_(nullptr),
              img_sub_(nullptr),
              num_features_(1000),
              timer_hz_(10.0),
              scale_factor_(1.2f),
              n_levels_(8),
              edge_threshold_(31),
              first_level_(0),
              wta_k_(2),
              score_type_(cv::ORB::HARRIS_SCORE),
              patch_size_(31),
              fast_threshold_(30)
        {
            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] OrbFeatureDetector constructor called", instance_name_.c_str());
        }

        // -----------------------------------------------------------------------------
        // Lifecycle-like methods
        // -----------------------------------------------------------------------------

        bool OrbFeatureDetector::configure(rclcpp_lifecycle::LifecycleNode *node,
                                           jwko::image_subscriber::ImageSubscriber *img_sub)
        {
            node_ = node;
            img_sub_ = img_sub;

            try
            {
                RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                            "[%s] Configuring OrbFeatureDetector...", instance_name_.c_str());

                setupParameters();

                // Publisher 생성 (configure 단계에서 생성, activate 전까지 비활성)
                feature_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(publish_topic_, 10);
                RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                            "[%s] Feature image publisher created on topic: %s",
                            instance_name_.c_str(), publish_topic_.c_str());

                initialize();

                RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                            "[%s] OrbFeatureDetector configured successfully", instance_name_.c_str());
                return true;
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(rclcpp::get_logger("OrbFeatureDetector"),
                             "[%s] Exception during configuration: %s",
                             instance_name_.c_str(), e.what());
                return false;
            }
        }

        bool OrbFeatureDetector::activate()
        {
            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] Activating OrbFeatureDetector...", instance_name_.c_str());

            // Publisher 활성화
            feature_pub_->on_activate();

            // 처리 타이머 시작
            auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(1.0 / timer_hz_));
            timer_ = node_->create_wall_timer(period, [this]()
                                              { process(); });

            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] OrbFeatureDetector activated (%.1f Hz)", instance_name_.c_str(), timer_hz_);
            return true;
        }

        bool OrbFeatureDetector::deactivate()
        {
            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] Deactivating OrbFeatureDetector...", instance_name_.c_str());

            // 타이머 중단
            if (timer_)
            {
                timer_->cancel();
                timer_.reset();
            }

            // Publisher 비활성화
            feature_pub_->on_deactivate();

            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] OrbFeatureDetector deactivated", instance_name_.c_str());
            return true;
        }

        bool OrbFeatureDetector::cleanup()
        {
            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] Cleaning up OrbFeatureDetector...", instance_name_.c_str());

            if (timer_)
            {
                timer_->cancel();
                timer_.reset();
            }

            feature_pub_.reset();
            orb_detector_.release();

            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] OrbFeatureDetector cleaned up", instance_name_.c_str());
            return true;
        }

        bool OrbFeatureDetector::shutdown()
        {
            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] Shutting down OrbFeatureDetector...", instance_name_.c_str());

            if (timer_)
            {
                timer_->cancel();
                timer_.reset();
            }

            feature_pub_.reset();
            orb_detector_.release();

            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] OrbFeatureDetector shut down", instance_name_.c_str());
            return true;
        }

        // -----------------------------------------------------------------------------
        // Private helpers
        // -----------------------------------------------------------------------------

        void OrbFeatureDetector::setupParameters()
        {
            // instance_name_ 을 prefix로 하여 파라미터 선언 (ImageSubscriber 패턴 동일)
            if (!node_->has_parameter(instance_name_ + "/num_features"))
            {
                num_features_ = node_->declare_parameter<int>(instance_name_ + "/num_features", 1000);
                timer_hz_ = node_->declare_parameter<double>(instance_name_ + "/timer_hz", 10.0);
                publish_topic_ = node_->declare_parameter<std::string>(
                    instance_name_ + "/publish_topic",
                    "visual_slam/" + instance_name_ + "/orb_features");
                frame_id_ = node_->declare_parameter<std::string>(
                    instance_name_ + "/frame_id",
                    instance_name_ + "_camera");
                scale_factor_ = static_cast<float>(
                    node_->declare_parameter<double>(instance_name_ + "/scale_factor", 1.2));
                n_levels_ = node_->declare_parameter<int>(instance_name_ + "/n_levels", 8);
                edge_threshold_ = node_->declare_parameter<int>(instance_name_ + "/edge_threshold", 31);
                first_level_ = node_->declare_parameter<int>(instance_name_ + "/first_level", 0);
                wta_k_ = node_->declare_parameter<int>(instance_name_ + "/wta_k", 2);
                score_type_ = node_->declare_parameter<int>(instance_name_ + "/score_type",
                                                            static_cast<int>(cv::ORB::HARRIS_SCORE));
                patch_size_ = node_->declare_parameter<int>(instance_name_ + "/patch_size", 31);
                fast_threshold_ = node_->declare_parameter<int>(instance_name_ + "/fast_threshold", 30);
            }
            else
            {
                num_features_ = node_->get_parameter(instance_name_ + "/num_features").as_int();
                timer_hz_ = node_->get_parameter(instance_name_ + "/timer_hz").as_double();
                publish_topic_ = node_->get_parameter(instance_name_ + "/publish_topic").as_string();
                frame_id_ = node_->get_parameter(instance_name_ + "/frame_id").as_string();
                scale_factor_ = static_cast<float>(
                    node_->get_parameter(instance_name_ + "/scale_factor").as_double());
                n_levels_ = node_->get_parameter(instance_name_ + "/n_levels").as_int();
                edge_threshold_ = node_->get_parameter(instance_name_ + "/edge_threshold").as_int();
                first_level_ = node_->get_parameter(instance_name_ + "/first_level").as_int();
                wta_k_ = node_->get_parameter(instance_name_ + "/wta_k").as_int();
                score_type_ = node_->get_parameter(instance_name_ + "/score_type").as_int();
                patch_size_ = node_->get_parameter(instance_name_ + "/patch_size").as_int();
                fast_threshold_ = node_->get_parameter(instance_name_ + "/fast_threshold").as_int();
            }

            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] num_features=%d  scale_factor=%.2f  n_levels=%d  "
                        "edge_threshold=%d  first_level=%d  wta_k=%d  "
                        "score_type=%d  patch_size=%d  fast_threshold=%d  "
                        "timer_hz=%.1f  topic=%s",
                        instance_name_.c_str(),
                        num_features_, scale_factor_, n_levels_,
                        edge_threshold_, first_level_, wta_k_,
                        score_type_, patch_size_, fast_threshold_,
                        timer_hz_, publish_topic_.c_str());
        }

        bool OrbFeatureDetector::initialize()
        {
            // compute()만 사용: 검출은 ComputeKeyPointsOctTree 에서 직접 FAST 수행
            orb_detector_ = cv::ORB::create(
                num_features_,
                scale_factor_,
                n_levels_,
                edge_threshold_,
                first_level_,
                wta_k_,
                static_cast<cv::ORB::ScoreType>(score_type_),
                patch_size_,
                fast_threshold_);

            RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                        "[%s] ORB detector created (max features: %d)",
                        instance_name_.c_str(), num_features_);
            return true;
        }

        // -----------------------------------------------------------------------------
        // ORB 검출 + 가시화 + 퍼블리시
        // -----------------------------------------------------------------------------

        void OrbFeatureDetector::process()
        {
            if (!img_sub_ || !img_sub_->isImageReady())
            {
                return;
            }

            cv::Mat color_image = img_sub_->getLatestImage();
            img_sub_->resetImageReady();

            if (color_image.empty())
            {
                RCLCPP_WARN(rclcpp::get_logger("OrbFeatureDetector"),
                            "[%s] Empty image, skipping ORB detection", instance_name_.c_str());
                return;
            }

            // 그레이스케일 변환
            cv::Mat gray;
            if (color_image.channels() == 3)
            {
                cv::cvtColor(color_image, gray, cv::COLOR_BGR2GRAY);
            }
            else
            {
                gray = color_image.clone();
            }

            // ORB-SLAM3 ComputeKeyPointsOctTree 방식:
            // FAST 셀 검출 → 레벨별 OctTree 균등화 → level-0 좌표로 변환
            std::vector<cv::KeyPoint> keypoints = ComputeKeyPointsOctTree(
                gray, num_features_, scale_factor_, n_levels_,
                edge_threshold_, fast_threshold_, 7);

            // 균등화된 키포인트로 ORB 디스크립터 계산
            cv::Mat descriptors;
            orb_detector_->compute(gray, keypoints, descriptors);

            RCLCPP_DEBUG(rclcpp::get_logger("OrbFeatureDetector"),
                         "[%s] Detected %zu ORB keypoints",
                         instance_name_.c_str(), keypoints.size());

            // 컬러 이미지 위에 키포인트를 점(point)으로 시각화
            cv::Mat viz_image = color_image.clone();

            // cv::drawKeypoints(
            //     color_image, keypoints, viz_image,
            //     cv::Scalar::all(-1),
            //     cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            for (const auto &kp : keypoints)
            {
                cv::circle(viz_image, kp.pt, 3, cv::Scalar(0, 255, 0), -1);
            }

            // 키포인트 수 텍스트 오버레이
            std::string text = "ORB keypoints: " + std::to_string(keypoints.size());
            cv::putText(viz_image, text,
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 255, 0), 2);

            // ROS 이미지 메시지로 변환 후 퍼블리시
            if (feature_pub_ && feature_pub_->is_activated())
            {
                auto msg = cv_bridge::CvImage(
                               std_msgs::msg::Header(),
                               sensor_msgs::image_encodings::BGR8,
                               viz_image)
                               .toImageMsg();
                msg->header.stamp = node_->now();
                msg->header.frame_id = frame_id_;
                feature_pub_->publish(*msg);
            }
        }

    } // namespace orb_feature_detector
} // namespace jwko
