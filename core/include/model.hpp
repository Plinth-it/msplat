#ifndef MODEL_H
#define MODEL_H

#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"
#include <cstdint>
#include <utility>
#include <vector>

int numShBases(int degree);
float psnr(const MTensor& rendered, const MTensor& gt);
float l1_loss(const MTensor& rendered, const MTensor& gt);
void quantizeRenderedForEval(MTensor& rendered);

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs, int growthStopIter = 15000,
        int maxSplats = 10000000, float growthSelectFraction = 0.25f,
        float opacityDecay = 0.004f, float scaleDecay = 0.002f,
        float meanNoiseWeight = 50.0f,
        float lrMean = 2e-5f, float lrMeanEnd = 2e-7f,
        float lrScale = 7e-3f, float lrScaleEnd = 5e-3f,
        float lrRotation = 0.002f, float lrCoeffsDc = 2e-3f,
        float lrCoeffsShScale = 10.0f, float lrOpacity = 0.012f,
        float randomInitSceneScale = 0.0f, bool reduceSecondMoment = false,
        uint32_t randomSeed = 42,
        const float* bgColor = nullptr,
        bool renderMip = false);

  ~Model(){ releaseOptimizers(); }

  void setupOptimizers();
  void releaseOptimizers();

  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step, int phaseStep = -1, int phaseTotal = -1);
  float prepareBrushRefineFlags(int step, int checkScreen, bool allowGrowth, float cullCenter[3]);
  void applyRefineDecay(int step);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  void saveLodPly(const std::string &filename, int step, int64_t targetCount);
  std::vector<float> computePupLodScores(std::vector<Camera> &cams);
  void decimateToLod(int64_t targetCount);
  void decimateToLod(int64_t targetCount, const std::vector<float> &scores);
  void saveSplat(const std::string &filename);
  int loadPly(const std::string &filename);
  void saveCheckpoint(const std::string &filename, int step);
  int loadCheckpoint(const std::string &filename);
  struct CamSetup {
    float fx, fy, cx, cy;
    int height, width, degree, degreesToUse;
    std::tuple<int,int,int> tileBounds;
    float cam_pos[3];
  };
  CamSetup prepareCam(Camera& cam, int step, int forcedDownscale = 0);
  void fullIteration(Camera& cam, int step, MTensor &gtPacked,
                     bool useLossMask, float lossMaskMean,
                     bool useAlphaLoss, float matchAlphaWeight,
                     const float *stepBgColor, bool compositeGt,
                     float ssimWeight, float lpipsLossWeight,
                     int forcedDownscale = 0);
  MTensor render(Camera& cam, int step, const float *bgColorOverride = nullptr);

  MTensor means;
  MTensor scales;
  MTensor quats;
  MTensor featuresDc;
  MTensor featuresRest;
  MTensor opacities;

  static constexpr int N_ADAM_GROUPS = 6;
  MTensor adam_exp_avg[N_ADAM_GROUPS];
  MTensor adam_exp_avg_sq[N_ADAM_GROUPS];
  int adam_step_count = 0;
  float adam_lr[N_ADAM_GROUPS] = {};
  float adam_beta1 = 0.9f, adam_beta2 = 0.999f, adam_eps = 1e-15f;
  float means_lr_init = 0, means_lr_final = 0;
  float baseMeansLrInit = 0, baseMeansLrFinal = 0;
  float currentMeanLrSceneScale = 1.0f;
  float meanNoiseMax = 1.0f;
  float scales_lr_init = 0, scales_lr_final = 0;
  float rotation_lr = 0, coeffs_dc_lr = 0, coeffs_rest_lr = 0, opacity_lr = 0;
  bool reduceSecondMoment = false;

  MTensor means_buf, scales_buf, quats_buf, featuresDc_buf, featuresRest_buf, opacities_buf;
  MTensor adam_exp_avg_buf[N_ADAM_GROUPS], adam_exp_avg_sq_buf[N_ADAM_GROUPS];
  int num_active = 0, buf_capacity = 0;
  void refreshViews();
  void ensureCapacity(int needed);
  void ensureLoadedShCapacity();
  void updateMeanLrSceneScale(float sceneScale, int scheduleStep = -1);
  void updateMeanLrSceneScaleFromActive(int scheduleStep = -1);

  MTensor densify_split_flag, densify_dup_flag;
  MTensor densify_split_prefix, densify_dup_prefix;
  MTensor densify_keep_flag, densify_keep_prefix;
  MTensor densify_block_totals;
  MTensor densify_compact_scratch;
  std::vector<float> refineScratchX, refineScratchY, refineScratchZ;
  std::vector<float> refineScratchWeights;
  std::vector<uint8_t> refineScratchPruned, refineScratchSelected;
  std::vector<std::pair<float, int>> refineScratchSampleKeys;

  MTensor radii;
  int lastHeight;
  int lastWidth;

  MTensor xysGradNorm;
  MTensor visCounts;
  MTensor max2DSize;

  MTensor backgroundColor;
  MTensor trainingBackgroundColor;
  MTensor window2d;  // SSIM window (11,11) f32

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  int maxSplats;
  float growthSelectFraction;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  float opacityDecay;
  float scaleDecay;
  float meanNoiseWeight;
  bool keepCrs;
  bool renderMip;

  float scale;
  float translation[3] = {};
};

#endif
