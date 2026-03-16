#include "orb_feature_detector.hpp"

#include <algorithm>
#include <chrono>
#include <string>

#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include "std_msgs/msg/header.hpp"

// =============================================================================
// ExtractorNode
// =============================================================================

void jwko::orb_feature_detector::ExtractorNode::DivideNode(
    ExtractorNode &n1, ExtractorNode &n2,
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

// =============================================================================
// compareNodes
// =============================================================================

bool jwko::orb_feature_detector::compareNodes(
    const std::pair<int, jwko::orb_feature_detector::ExtractorNode *> &e1,
    const std::pair<int, jwko::orb_feature_detector::ExtractorNode *> &e2)
{
    return e1.first < e2.first;
}

// =============================================================================
// DistributeOctTree  (ORB-SLAM3 원본 로직, standalone 버전)
// =============================================================================

std::vector<cv::KeyPoint> jwko::orb_feature_detector::DistributeOctTree(
    const std::vector<cv::KeyPoint> &vToDistributeKeys,
    const int &minX, const int &maxX,
    const int &minY, const int &maxY,
    const int &N, const int & /*level*/)
{
    using namespace jwko::orb_feature_detector;

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

// =============================================================================
// ComputeORBFeatures  (ORB-SLAM3 ORBextractor 방식 완전 이식)
//
// 원본과의 차이 수정:
//  1. ComputePyramid: copyMakeBorder(BORDER_REFLECT_101) 방식 패딩 피라미드
//  2. IC_Angle: intensity centroid 기반 키포인트 방향 계산
//  3. computeOrbDescriptor: bit_pattern_31_ 기반 직접 디스크립터 계산
//  4. 레벨별 GaussianBlur 후 디스크립터 계산 (level 좌표 그대로)
//  5. 디스크립터 계산 완료 후 level-0 좌표 변환 (기존 코드는 이 순서가 반대)
// =============================================================================

// ORB bit_pattern_31_ (512 포인트, ORB-SLAM3 원본 동일)
static const int kBitPattern31[256 * 4] = {
    8, -3, 9, 5, 4, 2, 7, -12, -11, 9, -8, 2, 7, -12, 12, -13,
    2, -13, 2, 12, 1, -7, 1, 6, -2, -10, -2, -4, -13, -13, -11, -8,
    -13, -3, -12, -9, 10, 4, 11, 9, -13, -8, -8, -9, -11, 7, -9, 12,
    7, 7, 12, 6, -4, -5, -3, 0, -13, 2, -12, -3, -9, 0, -7, 5,
    12, -6, 12, -1, -3, 6, -2, 12, -6, -13, -4, -8, 11, -13, 12, -8,
    4, 7, 5, 1, -1, 3, 0, 7, 7, -8, 7, 3, -4, 2, -3, 7,
    -7, -1, -6, 7, -13, -12, -8, -13, -7, -2, -6, -8, -8, 5, -6, -9,
    -5, -1, -4, 5, -13, 7, -12, 12, 6, -25, 7, -24, -3, -16, -2, -10,
    5, 1, 5, 5, 6, -5, 6, 5, 1, -11, 3, -11, -13, 3, -9, 4,
    -10, -14, -9, -9, 12, 9, 13, 6, -7, 1, -7, 11, -3, -14, -1, -10,
    5, -13, 7, -12, 2, -3, 3, 2, -5, -10, -4, -3, -4, -14, -4, -8,
    -1, -3, 0, 3, -9, 8, -8, 3, -4, -3, -3, -13, 0, 2, 1, 7,
    -3, 8, -3, -3, 2, 4, 3, 9, 9, -11, 10, -6, -9, 12, -8, 7,
    2, -13, 3, -9, -1, -13, 2, -6, 8, 3, 9, -3, 1, -8, 2, -7,
    5, 9, 6, 4, 0, -10, 0, -3, 12, 8, 13, 3, -5, -11, -5, -3,
    -7, -3, -7, 4, -5, -5, -5, 6, -11, 0, -11, 6, 2, -8, 2, 4,
    1, -13, 1, -8, 8, -13, 9, -12, 8, -1, 9, 4, -3, -2, -3, 6,
    -5, 3, -5, 11, 7, -4, 8, -1, 1, -4, 3, -4, 5, -2, 6, -1,
    -12, 6, -11, 1, -6, 2, -5, 9, -7, -5, -6, -3, -8, -7, -6, -4,
    2, -2, 4, -1, -13, -12, -13, 0, -6, 5, -5, 8, 5, -6, 6, -3,
    -2, 0, -2, 7, -1, -5, 0, 2, 7, -9, 9, -8, 3, -1, 5, -1,
    -12, 8, -11, 5, -10, 3, -8, 1, -9, -3, -8, 5, 8, -1, 9, 4,
    -12, -7, -10, -7, 5, 3, 6, 8, -2, -1, -1, 4, -9, -8, -9, -3,
    10, 5, 12, 5, -3, 9, -2, -3, 1, -3, 2, 1, -13, -8, -10, -8,
    -10, -5, -9, -2, 4, -4, 6, -2, 7, -8, 8, -9, -12, -1, -11, -1,
    2, 3, 3, -1, -6, 6, -5, 4, -3, -1, -2, 9, -11, -1, -9, 4,
    -6, -10, -5, -5, -12, -8, -9, -10, 3, -6, 5, -6, -4, -9, -2, -9,
    10, -7, 11, -12, -1, 3, 0, 6, 3, -3, 5, 0, -6, -1, -5, -10,
    -4, -12, -3, -9, 11, 7, 12, 11, 9, 4, 10, -3, 0, 0, 1, -5,
    -13, -7, -12, -12, 2, 7, 3, 11, -6, -6, -5, -4, -4, 3, -3, 8,
    5, -2, 7, -1, 7, 9, 8, 4, 8, -2, 9, 2, -3, -7, -3, 4,
    0, -5, 0, 4, -11, -3, -10, 4, -4, 3, -3, 8, -3, -9, -3, -2,
    -9, 0, -8, -5, 10, -9, 11, -6, -8, 4, -7, 9, 1, -2, 1, 4,
    -6, 1, -5, 6, -3, -3, -2, -8, 11, 10, 12, 5, -12, 10, -11, 5,
    -7, 11, -6, 7, -11, 3, -10, -1, 3, -9, 4, -5, 6, -10, 7, -8,
    11, -6, 12, -12, 7, 5, 8, 0, -2, -6, -1, -1, -13, 11, -12, 5,
    -6, -1, -5, 4, 7, 4, 8, -4, -13, 8, -11, 8, 1, -3, 2, 5,
    5, 7, 6, 2, 9, -8, 9, 0, 1, 3, 2, -4, -1, -5, -1, 6,
    4, 1, 4, 5, 4, -9, 4, -3, 1, -9, 2, -5, 2, -7, 3, -4,
    -7, 5, -6, 2, -7, -11, -6, -6, 4, -8, 5, -5, 0, 2, 1, -1,
    -12, 11, -11, 6, -2, -12, -2, 4, -1, -9, 0, -4, 8, 3, 9, -3,
    -5, -11, -5, -5, -9, 5, -8, 10, 7, -11, 8, -8, 0, -13, 0, -6,
    2, -11, 2, -5, -13, -7, -12, -12, 12, -4, 13, -1, -12, 0, -10, 0,
    -7, 7, -6, 12, -9, 3, -8, -2, 3, -5, 4, 0, -9, -11, -9, -4,
    4, -6, 4, 1, -6, -2, -5, 3, -12, -9, -11, -4, -7, 6, -6, 11,
    2, -13, 3, -9, -13, 2, -12, -3, -3, -10, -2, -5, 0, 7, 1, 12,
    -12, -3, -11, 2, 9, 1, 10, 6, -1, -6, 0, -1, -9, -1, -8, 4,
    3, -10, 4, -7, 12, 2, 13, 7, 12, -4, 13, 1, 0, -3, 1, 2,
    3, -7, 4, -4, -9, -11, -9, -3, 4, -7, 5, -3, -11, 2, -10, 7,
    -12, -4, -12, 4, 10, 1, 11, 5, -13, -1, -12, 4, 8, -9, 9, -13,
    -2, 6, -2, 12, -4, 7, -4, 13, -7, -5, -6, -9, -10, -2, -9, 3,
    -8, -4, -6, -4, 11, -1, 12, 4, -9, -6, -8, -1, -1, -5, 0, 3,
    -13, -2, -12, 3, -6, -10, -5, -7, -13, -8, -12, -3, 3, -4, 5, -3,
    7, -10, 8, -6, 0, -13, 0, -7, 2, -1, 3, 4, -2, -9, -2, -3,
    -13, 5, -12, 10, 3, 2, 4, 6, 12, -2, 12, 3, 10, 4, 11, 9,
    -6, -1, -5, 4, -9, 0, -7, -1, 3, -13, 5, -13, -12, 7, -10, 7,
    -7, -1, -7, 6, -6, -8, -4, -8, -6, 2, -5, 7, -3, 4, -2, -1,
    8, 7, 9, 2, -3, 0, -2, 5, -5, -13, -3, -13, -9, 3, -7, 3,
    -8, -3, -6, -3, -12, -11, -11, -6, 4, -9, 5, -5, 5, -1, 6, 4,
    5, -10, 6, -6, 1, -4, 2, 0, 11, -9, 12, -5, 3, -1, 3, 5,
    -6, 4, -5, 9, -11, 0, -10, 5, -13, -9, -11, -9, -9, -12, -8, -7,
    -2, 3, -1, 8, -12, -4, -10, -4, 5, -5, 6, -1, 12, 9, 13, 4,
    0, -7, 1, -3, 6, -3, 6, 2, 7, 5, 8, 9, -7, -8, -5, -8,
    2, -11, 3, -7, 4, -1, 5, 4, 12, 0, 12, 5, 12, -10, 13, -6};

static const int HALF_PATCH_SIZE = 15;

static float IC_Angle(const cv::Mat &image, cv::Point2f pt, const std::vector<int> &u_max)
{
    int m_01 = 0, m_10 = 0;
    const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];
    int step = static_cast<int>(image.step1());
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step];
            int val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }
    return cv::fastAtan2(static_cast<float>(m_01), static_cast<float>(m_10));
}

static void computeOrbDescriptor(const cv::KeyPoint &kpt,
                                 const cv::Mat &img,
                                 const cv::Point *pattern,
                                 uchar *desc)
{
    const float factorPI = static_cast<float>(CV_PI / 180.0);
    float angle = kpt.angle * factorPI;
    float a = cosf(angle), b = sinf(angle);
    const uchar *center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = static_cast<int>(img.step);

#define GET_VALUE(idx)                                               \
    center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + \
           cvRound(pattern[idx].x * a - pattern[idx].y * b)]

    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0);
        t1 = GET_VALUE(1);
        val = (t0 < t1);
        t0 = GET_VALUE(2);
        t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;
        desc[i] = static_cast<uchar>(val);
    }
#undef GET_VALUE
}

void jwko::orb_feature_detector::ComputeORBFeatures(
    const cv::Mat &image,
    int nfeatures, float scaleFactor, int nlevels,
    int edgeThreshold, int iniThFAST, int minThFAST,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &descriptors)
{
    using namespace jwko::orb_feature_detector;

    // --- 스케일 팩터 사전 계산 ---
    std::vector<float> mvScaleFactor(nlevels);
    std::vector<float> mvInvScaleFactor(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvInvScaleFactor[0] = 1.0f;
    for (int i = 1; i < nlevels; ++i)
    {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
    }

    // --- umax 테이블 (IC_Angle용) ---
    std::vector<int> umax(HALF_PATCH_SIZE + 1);
    {
        int vmax = cvFloor(HALF_PATCH_SIZE * std::sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * std::sqrt(2.f) / 2);
        const double hp2 = static_cast<double>(HALF_PATCH_SIZE) * HALF_PATCH_SIZE;
        for (int v = 0; v <= vmax; ++v)
            umax[v] = cvRound(std::sqrt(hp2 - static_cast<double>(v) * v));
        for (int v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0++;
        }
    }

    // --- bit_pattern_31_ 포인터 ---
    const cv::Point *pattern = reinterpret_cast<const cv::Point *>(kBitPattern31);

    // --- 레벨별 목표 피처 수 ---
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

    // --- ORB-SLAM3 방식 피라미드 구축: EDGE_THRESHOLD 패딩 + BORDER_REFLECT_101 ---
    std::vector<cv::Mat> mvImagePyramid(nlevels);
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        cv::Size sz(cvRound(static_cast<float>(image.cols) * scale),
                    cvRound(static_cast<float>(image.rows) * scale));
        cv::Size wholeSize(sz.width + edgeThreshold * 2, sz.height + edgeThreshold * 2);
        cv::Mat temp(wholeSize, image.type());
        mvImagePyramid[level] = temp(cv::Rect(edgeThreshold, edgeThreshold, sz.width, sz.height));

        if (level != 0)
        {
            cv::resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);
            cv::copyMakeBorder(mvImagePyramid[level], temp,
                               edgeThreshold, edgeThreshold, edgeThreshold, edgeThreshold,
                               cv::BORDER_REFLECT_101 | cv::BORDER_ISOLATED);
        }
        else
        {
            cv::copyMakeBorder(image, temp,
                               edgeThreshold, edgeThreshold, edgeThreshold, edgeThreshold,
                               cv::BORDER_REFLECT_101);
        }
    }

    // --- 레벨별 키포인트 검출 (OctTree 균등화) + 방향 계산 ---
    std::vector<std::vector<cv::KeyPoint>> allKeypoints(nlevels);
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

        // OctTree 균등 분포
        allKeypoints[level] = DistributeOctTree(
            vToDistributeKeys,
            minBorderX, maxBorderX, minBorderY, maxBorderY,
            mnFeaturesPerLevel[level], level);

        // border offset + octave + size 설정 (아직 level 좌표)
        const int scaledPatchSize = static_cast<int>(31 * mvScaleFactor[level]);
        for (auto &kp : allKeypoints[level])
        {
            kp.pt.x += minBorderX;
            kp.pt.y += minBorderY;
            kp.octave = level;
            kp.size = static_cast<float>(scaledPatchSize);
        }

        // IC_Angle: 방향 계산 (level 이미지 좌표 기준)
        for (auto &kp : allKeypoints[level])
            kp.angle = IC_Angle(mvImagePyramid[level], kp.pt, umax);
    }

    // --- 레벨별 GaussianBlur → 직접 디스크립터 계산 → level-0 좌표 변환 ---
    int totalKps = 0;
    for (int level = 0; level < nlevels; ++level)
        totalKps += static_cast<int>(allKeypoints[level].size());

    keypoints.clear();
    keypoints.reserve(totalKps);
    descriptors = cv::Mat::zeros(totalKps, 32, CV_8UC1);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        auto &levelKps = allKeypoints[level];
        if (levelKps.empty())
            continue;

        // GaussianBlur (ORB-SLAM3 동일 파라미터)
        cv::Mat workingMat = mvImagePyramid[level].clone();
        cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        // 각 키포인트 디스크립터 계산 (level 좌표 기준)
        for (int i = 0; i < static_cast<int>(levelKps.size()); ++i)
            computeOrbDescriptor(levelKps[i], workingMat, pattern,
                                 descriptors.ptr(offset + i));

        // 이제 level-0 좌표로 변환
        if (level != 0)
        {
            const float scale = mvScaleFactor[level];
            for (auto &kp : levelKps)
                kp.pt *= scale;
        }

        keypoints.insert(keypoints.end(), levelKps.begin(), levelKps.end());
        offset += static_cast<int>(levelKps.size());
    }
}

// =============================================================================
// OrbFeatureDetector constructor
// =============================================================================

jwko::orb_feature_detector::OrbFeatureDetector::OrbFeatureDetector(
    const std::string &instance_name)
    : instance_name_(instance_name),
      node_(nullptr),
      img_sub_(nullptr),
      num_features_(1000),
      timer_hz_(10.0),
      scale_factor_(1.2f),
      n_levels_(8),
      edge_threshold_(19),
      fast_threshold_(20),
      min_kp_dist_(0.0f)
{
    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] OrbFeatureDetector constructor called", instance_name_.c_str());
}

// =============================================================================
// Lifecycle-like methods
// =============================================================================

bool jwko::orb_feature_detector::OrbFeatureDetector::configure(
    rclcpp_lifecycle::LifecycleNode *node,
    jwko::image_subscriber::ImageSubscriber *img_sub)
{
    node_ = node;
    img_sub_ = img_sub;

    try
    {
        RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                    "[%s] Configuring OrbFeatureDetector...", instance_name_.c_str());

        setupParameters();

        // Publisher 생성 (configure 단계, activate 전까지 비활성)
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

bool jwko::orb_feature_detector::OrbFeatureDetector::activate()
{
    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] Activating OrbFeatureDetector...", instance_name_.c_str());

    feature_pub_->on_activate();

    auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / timer_hz_));
    timer_ = node_->create_wall_timer(period, [this]()
                                      { process(); });

    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] OrbFeatureDetector activated (%.1f Hz)", instance_name_.c_str(), timer_hz_);
    return true;
}

bool jwko::orb_feature_detector::OrbFeatureDetector::deactivate()
{
    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] Deactivating OrbFeatureDetector...", instance_name_.c_str());

    if (timer_)
    {
        timer_->cancel();
        timer_.reset();
    }

    feature_pub_->on_deactivate();

    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] OrbFeatureDetector deactivated", instance_name_.c_str());
    return true;
}

bool jwko::orb_feature_detector::OrbFeatureDetector::cleanup()
{
    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] Cleaning up OrbFeatureDetector...", instance_name_.c_str());

    if (timer_)
    {
        timer_->cancel();
        timer_.reset();
    }

    feature_pub_.reset();

    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] OrbFeatureDetector cleaned up", instance_name_.c_str());
    return true;
}

bool jwko::orb_feature_detector::OrbFeatureDetector::shutdown()
{
    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] Shutting down OrbFeatureDetector...", instance_name_.c_str());

    if (timer_)
    {
        timer_->cancel();
        timer_.reset();
    }

    feature_pub_.reset();

    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] OrbFeatureDetector shut down", instance_name_.c_str());
    return true;
}

// =============================================================================
// Private helpers
// =============================================================================

void jwko::orb_feature_detector::OrbFeatureDetector::setupParameters()
{
    if (!node_->has_parameter(instance_name_ + "/num_features"))
    {
        num_features_ = node_->declare_parameter<int>(instance_name_ + "/num_features", 1000);
        timer_hz_ = node_->declare_parameter<double>(instance_name_ + "/timer_hz", 10.0);
        publish_topic_ = node_->declare_parameter<std::string>(
            instance_name_ + "/publish_topic",
            "visual_slam/" + instance_name_ + "/orb_features");
        frame_id_ = node_->declare_parameter<std::string>(
            instance_name_ + "/frame_id", instance_name_ + "_camera");
        scale_factor_ = static_cast<float>(
            node_->declare_parameter<double>(instance_name_ + "/scale_factor", 1.2));
        n_levels_ = node_->declare_parameter<int>(instance_name_ + "/n_levels", 8);
        edge_threshold_ = node_->declare_parameter<int>(instance_name_ + "/edge_threshold", 19);
        fast_threshold_ = node_->declare_parameter<int>(instance_name_ + "/fast_threshold", 20);
        min_kp_dist_ = static_cast<float>(
            node_->declare_parameter<double>(instance_name_ + "/min_kp_dist", 0.0));
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
        fast_threshold_ = node_->get_parameter(instance_name_ + "/fast_threshold").as_int();
        min_kp_dist_ = static_cast<float>(
            node_->get_parameter(instance_name_ + "/min_kp_dist").as_double());
    }

    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] num_features=%d  scale_factor=%.2f  n_levels=%d  "
                "edge_threshold=%d  fast_threshold=%d  min_kp_dist=%.1f  "
                "timer_hz=%.1f  topic=%s",
                instance_name_.c_str(),
                num_features_, scale_factor_, n_levels_,
                edge_threshold_, fast_threshold_, min_kp_dist_,
                timer_hz_, publish_topic_.c_str());
}

bool jwko::orb_feature_detector::OrbFeatureDetector::initialize()
{
    RCLCPP_INFO(rclcpp::get_logger("OrbFeatureDetector"),
                "[%s] ORB extractor initialized (ORB-SLAM3 방식, max features: %d)",
                instance_name_.c_str(), num_features_);
    return true;
}

// =============================================================================
// ORB 검출 + 가시화 + 퍼블리시
// =============================================================================

void jwko::orb_feature_detector::OrbFeatureDetector::process()
{
    if (!img_sub_ || !img_sub_->isImageReady())
        return;

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
        cv::cvtColor(color_image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = color_image.clone();

    // ORB-SLAM3 완전 이식:
    // REFLECT_101 패딩 피라미드 → FAST+OctTree → IC_Angle → GaussianBlur → bit_pattern 디스크립터
    // → 디스크립터 계산 완료 후 level-0 좌표 변환
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    ComputeORBFeatures(
        gray, num_features_, scale_factor_, n_levels_,
        edge_threshold_, fast_threshold_, 7,
        keypoints, descriptors);

    // (선택) 크로스-레벨 전역 최소 거리 필터: min_kp_dist_ > 0 이면 추가 스파스화
    if (min_kp_dist_ > 0.0f)
    {
        std::sort(keypoints.begin(), keypoints.end(),
                  [](const cv::KeyPoint &a, const cv::KeyPoint &b)
                  { return a.response > b.response; });

        const int gW = static_cast<int>(gray.cols / min_kp_dist_) + 1;
        const int gH = static_cast<int>(gray.rows / min_kp_dist_) + 1;
        std::vector<bool> occupied(gW * gH, false);
        std::vector<int> keep_idx;
        keep_idx.reserve(keypoints.size());

        for (int i = 0; i < static_cast<int>(keypoints.size()); ++i)
        {
            int gx = std::min(static_cast<int>(keypoints[i].pt.x / min_kp_dist_), gW - 1);
            int gy = std::min(static_cast<int>(keypoints[i].pt.y / min_kp_dist_), gH - 1);
            if (!occupied[gy * gW + gx])
            {
                occupied[gy * gW + gx] = true;
                keep_idx.push_back(i);
            }
        }

        std::vector<cv::KeyPoint> filtered_kps;
        cv::Mat filtered_desc(static_cast<int>(keep_idx.size()), 32, CV_8UC1);
        filtered_kps.reserve(keep_idx.size());
        for (int k = 0; k < static_cast<int>(keep_idx.size()); ++k)
        {
            filtered_kps.push_back(keypoints[keep_idx[k]]);
            descriptors.row(keep_idx[k]).copyTo(filtered_desc.row(k));
        }
        keypoints = std::move(filtered_kps);
        descriptors = std::move(filtered_desc);
    }

    RCLCPP_DEBUG(rclcpp::get_logger("OrbFeatureDetector"),
                 "[%s] Detected %zu ORB keypoints",
                 instance_name_.c_str(), keypoints.size());

    // 시각화: ORB-SLAM3 스타일 (5px 박스 + 2px 점)
    cv::Mat viz_image = color_image.clone();
    const float r = 5.0f;
    for (const auto &kp : keypoints)
    {
        cv::Point2f pt1(kp.pt.x - r, kp.pt.y - r);
        cv::Point2f pt2(kp.pt.x + r, kp.pt.y + r);
        cv::rectangle(viz_image, pt1, pt2, cv::Scalar(0, 255, 0));
        cv::circle(viz_image, kp.pt, 2, cv::Scalar(0, 255, 0), -1);
    }

    std::string text = "ORB keypoints: " + std::to_string(keypoints.size());
    cv::putText(viz_image, text,
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);

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
