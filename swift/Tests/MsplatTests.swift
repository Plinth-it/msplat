import XCTest
import Msplat

final class MsplatTests: XCTestCase {

    static let gardenPath = "../datasets/mipnerf360/garden"

    func testConfigDefaults() {
        let config = TrainingConfig()
        XCTAssertEqual(config.iterations, 30_000)
        XCTAssertEqual(config.shDegree, 3)
        XCTAssertEqual(config.shDegreeInterval, 1)
        XCTAssertEqual(config.ssimWeight, 0.2, accuracy: 0.001)
        XCTAssertEqual(config.numDownscales, 0)
        XCTAssertEqual(config.refineEvery, 200)
        XCTAssertEqual(config.warmupLength, 0)
        XCTAssertEqual(config.resetAlphaEvery, 0)
        XCTAssertEqual(config.densifyGradThresh, 0.0020, accuracy: 0.00001)
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
