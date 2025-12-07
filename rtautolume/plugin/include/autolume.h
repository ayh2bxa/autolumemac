#include <torch/torch.h>
#include <torch/script.h>
#include "defines.h"
#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <Accelerate/Accelerate.h>

using namespace std;
class Autolume
{
public:
    Autolume();
    ~Autolume();

    // Must be called after JUCE initialization (e.g., in prepareToPlay)
    void initialize();

    // Load model from file path (called from GUI thread)
    bool loadModel(const std::string& path);

    // Check if renderer is ready for use
    bool isReady() const {
        return isInitialized.load(std::memory_order_acquire) &&
               modelLoaded.load(std::memory_order_acquire);
    }

    void processAudio(float val);

    // Called from GUI thread: request inference to run
    void requestInference();

    // Called from GUI thread: copy latest 512x512 RGB frame into dest
    bool getLatestFrame (uint8_t* dest, size_t numBytes);

    // Noise strength control (called from GUI thread)
    void setNoiseStrength(float value);
    float getNoiseStrength() const;

    // Latent control (called from GUI thread)
    void setLatentSpeed(float value);
    float getLatentSpeed() const;

private:
    // Find and cache noise_strength parameters from model
    void findNoiseStrengthParameters();
    // Inference thread
    void inferenceThreadLoop();
    void runInference();

    // Model
    torch::jit::script::Module model;
    torch::Device device{torch::kCPU};  // Start with CPU, switch to MPS on inference thread
    torch::Tensor inputTensor;
    vector<torch::jit::IValue> inputs;
    std::string modelPath;  // Path to loaded model

    // Noise strength parameters (cached for real-time control)
    std::vector<torch::Tensor> noiseStrengthParams;

    // Latent control state
    atomic<float> latentX{0.0f};
    atomic<float> latentY{0.0f};
    atomic<float> latentSpeed{0.25f};
    std::chrono::steady_clock::time_point lastLatentUpdate;

    // Audio thread data
    array<float, Constants::max_buf_size> in_buf;
    int rp = 0;
    int cnt = 0;

    // Shared between audio thread (write) and inference thread (read)
    alignas(64) atomic<bool> inputReady{false};
    array<float, Constants::nfft> ordered_in_buf;  // Written by audio thread
    array<float, Constants::nfft> inference_input_buf;  // FFT magnitude output for inference

    // FFT setup (vDSP Accelerate framework)
    FFTSetup fftSetup;
    DSPSplitComplex fftSplit;
    array<float, Constants::nfft / 2> fftReal;
    array<float, Constants::nfft / 2> fftImag;
    vDSP_Length fftLog2n;

    // Double buffer for frames: written by inference thread, read by GUI thread
    array<uint8_t, Constants::frameBytes> frameBuffer[2];
    atomic<int> readableFrameIndex{0};  // Which buffer is ready for GUI to read
    int writeFrameIndex = 1;  // Which buffer inference thread writes to
    mutex frameMutex;  // Protects frame swap

    // Thread control
    atomic<bool> shouldExit{false};
    atomic<bool> isInitialized{false};
    atomic<bool> modelLoaded{false};
    atomic<bool> mpsInitialized{false};
    atomic<bool> inferenceRunning{false};  // Prevent overlapping inference calls
    atomic<bool> inferenceRequested{false};  // Signal from GUI thread to inference thread
    thread inferenceThread;
};
