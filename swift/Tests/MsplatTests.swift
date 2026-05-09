import XCTest
import Msplat

final class MsplatTests: XCTestCase {

    static let gardenPath = "../datasets/mipnerf360/garden"
    private static let onePixelPNG: Data = {
        guard let data = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/l+kC6wAAAABJRU5ErkJggg==") else {
            fatalError("Invalid embedded PNG fixture")
        }
        return data
    }()

    func testConfigDefaults() {
        let config = TrainingConfig()
        XCTAssertEqual(config.iterations, 30_000)
        XCTAssertEqual(config.shDegree, 3)
        XCTAssertEqual(config.shDegreeInterval, 0)
        XCTAssertEqual(config.ssimWeight, 0.2, accuracy: 0.001)
        XCTAssertEqual(config.numDownscales, 0)
        XCTAssertEqual(config.refineEvery, 200)
        XCTAssertEqual(config.warmupLength, 0)
        XCTAssertEqual(config.resetAlphaEvery, 0)
        XCTAssertEqual(config.densifyGradThresh, 0.0025, accuracy: 0.00001)
        XCTAssertEqual(config.stopScreenSizeAt, 15_000)
        XCTAssertEqual(config.splitScreenSize, 0.25, accuracy: 0.00001)
        XCTAssertEqual(config.lrMean, 0.00002, accuracy: 0.0000001)
        XCTAssertEqual(config.lrMeanEnd, 0.0000002, accuracy: 0.00000001)
        XCTAssertEqual(config.lrScale, 0.007, accuracy: 0.000001)
        XCTAssertEqual(config.lrScaleEnd, 0.005, accuracy: 0.000001)
        XCTAssertEqual(config.lrCoeffsDc, 0.002, accuracy: 0.000001)
        XCTAssertEqual(config.lrOpacity, 0.012, accuracy: 0.000001)
        XCTAssertEqual(config.bgColor.0, 0.0, accuracy: 0.00001)
        XCTAssertEqual(config.bgColor.1, 0.0, accuracy: 0.00001)
        XCTAssertEqual(config.bgColor.2, 0.0, accuracy: 0.00001)
        XCTAssertEqual(config.lpipsLossWeight, 0.0, accuracy: 0.001)
        XCTAssertEqual(config.randomInitSceneScale, 0.0, accuracy: 0.001)
        XCTAssertFalse(config.reduceSecondMoment)
        XCTAssertEqual(config.imagePrefetchWorkers, 2)
    }

    func testLoadDataset() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0,
            evalMode: true,
            testEvery: 8
        )
        XCTAssertGreaterThan(dataset.numTrain, 0)
        XCTAssertGreaterThan(dataset.numTest, 0)
    }

    func testDatasetFiltersNonFinitePointCloudRows() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("msplat-swift-loader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.onePixelPNG.write(to: root.appendingPathComponent("image.png"))
        try """
        ply
        format ascii 1.0
        element vertex 4
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        0 0 0 255 255 255
        nan 0 0 255 0 0
        1 2 3 0 255 0
        0 inf 0 0 0 255
        """.write(to: root.appendingPathComponent("points3D.ply"), atomically: true, encoding: .utf8)
        try """
        {
          "w": 1,
          "h": 1,
          "fl_x": 1.0,
          "fl_y": 1.0,
          "cx": 0.5,
          "cy": 0.5,
          "frames": [
            {
              "file_path": "image.png",
              "transform_matrix": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
              ]
            }
          ]
        }
        """.write(to: root.appendingPathComponent("transforms.json"), atomically: true, encoding: .utf8)

        let dataset = GaussianDataset(path: root.path)

        XCTAssertEqual(dataset.numTrain, 1)
        XCTAssertEqual(dataset.initialPointCount, 2)
    }

    func testTrainShort() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 10
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)

        for _ in 0..<10 {
            let stats = trainer.step()
            XCTAssertGreaterThan(stats.splatCount, 0)
        }

        XCTAssertEqual(trainer.iteration, 10)
        XCTAssertGreaterThan(trainer.splatCount, 100_000)
    }

    func testRender() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 5
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)
        for _ in 0..<5 { trainer.step() }

        let rendered = trainer.render(cameraIndex: 0)
        XCTAssertGreaterThan(rendered.width, 0)
        XCTAssertGreaterThan(rendered.height, 0)
        XCTAssertEqual(rendered.pixels.count, rendered.width * rendered.height * 3)
    }

    func testRenderFromPoseToBufferDimensionQuery() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 1
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)
        let pose = dataset.cameraPose(at: 0)
        var width: Int32 = 0
        var height: Int32 = 0

        trainer.renderFromPoseToBuffer(
            camToWorld: pose,
            rgba: nil,
            width: &width,
            height: &height
        )

        XCTAssertGreaterThan(width, 0)
        XCTAssertGreaterThan(height, 0)
    }

    func testExportPly() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 5
        config.numDownscales = 0

        let trainer = GaussianTrainer(dataset: dataset, config: config)
        for _ in 0..<5 { trainer.step() }

        let tmpPath = NSTemporaryDirectory() + "msplat_test_export.ply"
        trainer.exportPly(to: tmpPath)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tmpPath))

        let fileSize = try FileManager.default.attributesOfItem(atPath: tmpPath)[.size] as! Int
        XCTAssertGreaterThan(fileSize, 0)

        try FileManager.default.removeItem(atPath: tmpPath)
    }

    func testLoadPlyReturnsIteration() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 1
        config.numDownscales = 0

        let tmpPath = NSTemporaryDirectory() + "msplat_test_load.ply"
        do {
            let source = GaussianTrainer(dataset: dataset, config: config)
            source.step()
            source.exportPly(to: tmpPath)

            let loaded = GaussianTrainer(dataset: dataset, config: config)
            let iteration = loaded.loadPly(from: tmpPath)
            XCTAssertEqual(iteration, 1)
            XCTAssertEqual(loaded.iteration, 1)
            XCTAssertEqual(loaded.splatCount, source.splatCount)
        }

        try FileManager.default.removeItem(atPath: tmpPath)
    }

    func testTrainerDeinitDoesNotCleanupSharedMetalState() throws {
        let dataset = GaussianDataset(
            path: Self.gardenPath,
            downscaleFactor: 4.0
        )
        var config = TrainingConfig()
        config.iterations = 2
        config.numDownscales = 0

        let survivor = GaussianTrainer(dataset: dataset, config: config)
        do {
            let shortLived = GaussianTrainer(dataset: dataset, config: config)
            XCTAssertGreaterThan(shortLived.step().splatCount, 0)
        }

        XCTAssertGreaterThan(survivor.step().splatCount, 0)
        XCTAssertGreaterThan(survivor.render(cameraIndex: 0).pixels.count, 0)
    }
}
