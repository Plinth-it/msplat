import MsplatCore

/// Configuration for Gaussian splatting training.
public struct TrainingConfig {
    public var iterations: Int32 = 30_000
    public var shDegree: Int32 = 3
    public var shDegreeInterval: Int32 = 1
    public var ssimWeight: Float = 0.2
    public var numDownscales: Int32 = 0
    public var resolutionSchedule: Int32 = 3_000
    public var refineEvery: Int32 = 200
    public var warmupLength: Int32 = 0
    public var resetAlphaEvery: Int32 = 0
    public var densifyGradThresh: Float = 0.0020
    public var densifySizeThresh: Float = 0.01
    public var stopScreenSizeAt: Int32 = 15_000
    public var growthStopIter: Int32 = 15_000
    public var maxSplats: Int32 = 10_000_000
    public var growthSelectFraction: Float = 0.25
    public var splitScreenSize: Float = 0.25
    public var matchAlphaWeight: Float = 0.1
    public var lpipsLossWeight: Float = 0.0
    public var opacityDecay: Float = 0.004
    public var scaleDecay: Float = 0.002
    public var meanNoiseWeight: Float = 50.0
    public var lrMean: Float = 0.00002
    public var lrMeanEnd: Float = 0.0000002
    public var lrScale: Float = 0.007
    public var lrScaleEnd: Float = 0.005
    public var lrRotation: Float = 0.002
    public var lrCoeffsDc: Float = 0.002
    public var lrCoeffsShScale: Float = 10.0
    public var lrOpacity: Float = 0.012
    /// Scene scale for random init when no point cloud exists. Use 0 to estimate from cameras.
    public var randomInitSceneScale: Float = 0.0
    /// Use Brush-style scalar second moment for SH Adam updates.
    public var reduceSecondMoment: Bool = false
    public var keepCrs: Bool = false
    /// Use MIP splatting opacity compensation during training and rendering.
    public var renderMip: Bool = false
    public var downscaleFactor: Float = 1.0
    /// Background color as (R, G, B) in [0, 1]. Defaults to Brush-compatible black.
    public var bgColor: (Float, Float, Float) = (0.0, 0.0, 0.0)
    public var backgroundNoiseStrength: Float = 0.1

    public init() {}

    func toC() -> MsplatConfig {
        var c = msplat_default_config()
        c.iterations = iterations
        c.shDegree = shDegree
        c.shDegreeInterval = shDegreeInterval
        c.ssimWeight = ssimWeight
        c.numDownscales = numDownscales
        c.resolutionSchedule = resolutionSchedule
        c.refineEvery = refineEvery
        c.warmupLength = warmupLength
        c.resetAlphaEvery = resetAlphaEvery
        c.densifyGradThresh = densifyGradThresh
        c.densifySizeThresh = densifySizeThresh
        c.stopScreenSizeAt = stopScreenSizeAt
        c.growthStopIter = growthStopIter
        c.maxSplats = maxSplats
        c.growthSelectFraction = growthSelectFraction
        c.splitScreenSize = splitScreenSize
        c.matchAlphaWeight = matchAlphaWeight
        c.lpipsLossWeight = lpipsLossWeight
        c.opacityDecay = opacityDecay
        c.scaleDecay = scaleDecay
        c.meanNoiseWeight = meanNoiseWeight
        c.lrMean = lrMean
        c.lrMeanEnd = lrMeanEnd
        c.lrScale = lrScale
        c.lrScaleEnd = lrScaleEnd
        c.lrRotation = lrRotation
        c.lrCoeffsDc = lrCoeffsDc
        c.lrCoeffsShScale = lrCoeffsShScale
        c.lrOpacity = lrOpacity
        c.randomInitSceneScale = randomInitSceneScale
        c.reduceSecondMoment = reduceSecondMoment
        c.keepCrs = keepCrs
        c.renderMip = renderMip
        c.downscaleFactor = downscaleFactor
        c.bgColor = (bgColor.0, bgColor.1, bgColor.2)
        c.backgroundNoiseStrength = backgroundNoiseStrength
        return c
    }
}
