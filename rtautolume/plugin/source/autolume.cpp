#include "autolume.h"
#include <chrono>
#include <cmath>

Autolume::Autolume() {
    // Initialize frame buffers to black
    frameBuffer[0].fill(0);
    frameBuffer[1].fill(0);
    inference_input_buf.fill(0.0f);

    // Initialize FFT setup (512-point FFT, log2(512) = 9)
    fftLog2n = static_cast<vDSP_Length>(std::log2(Constants::nfft));
    fftSetup = vDSP_create_fftsetup(fftLog2n, FFT_RADIX2);

    // Setup split complex buffer
    fftSplit.realp = fftReal.data();
    fftSplit.imagp = fftImag.data();

    std::cout << "Autolume: FFT setup initialized (log2n=" << fftLog2n << ")" << std::endl;
}

void Autolume::initialize() {
    // Prevent double initialization
    if (isInitialized.load(std::memory_order_acquire)) {
        std::cout << "Autolume: Already initialized, skipping" << std::endl;
        return;
    }

    std::cout << "Autolume: Starting deferred initialization..." << std::endl;

    try {
        // Load model
        std::cout << "Autolume: Loading model..." << std::endl;
        model = torch::jit::load("/Users/anthonyhong/Desktop/self/code/autolume_/stylegan2_notrain.pt");
        model.eval();

        // Prepare input tensor
        std::cout << "Autolume: Creating input tensor..." << std::endl;
        inputTensor = torch::empty({1, Constants::nfft}, torch::kFloat32);
        inputs.reserve(1);
        inputs.clear();
        inputs.emplace_back(inputTensor);

        // Find and cache noise_strength parameters
        findNoiseStrengthParameters();

        std::cout << "Autolume: Model initialized successfully" << std::endl;

        // Mark as initialized before starting thread
        isInitialized.store(true, std::memory_order_release);

        // Start inference thread AFTER everything is initialized
        shouldExit.store(false, std::memory_order_release);
        inferenceThread = std::thread(&Autolume::inferenceThreadLoop, this);

        std::cout << "Autolume: Inference thread started" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Autolume: ERROR during initialization: " << e.what() << std::endl;
        // Don't start the thread if initialization failed
    }
}

Autolume::~Autolume() {
    // Signal thread to exit
    shouldExit.store(true, std::memory_order_release);

    // Wait for thread to finish
    if (inferenceThread.joinable()) {
        inferenceThread.join();
    }

    // Clean up FFT setup
    if (fftSetup) {
        vDSP_destroy_fftsetup(fftSetup);
    }
}

void Autolume::processAudio(float val) {
    // Audio thread: accumulate samples into circular buffer
    in_buf[rp] = val;
    rp = (rp + 1) & (Constants::max_buf_size - 1);
    cnt++;

    // Every nfft samples, copy to ordered_in_buf and signal inference thread
    if (cnt >= Constants::nfft) {
        cnt = 0;

        // Copy samples in order
        for (size_t i = 0; i < Constants::nfft; i++) {
            ordered_in_buf[i] = in_buf[(rp + i - Constants::nfft + Constants::max_buf_size) & (Constants::max_buf_size - 1)];
        }

        // Signal that new input is ready (lock-free atomic flag)
        inputReady.store(true, std::memory_order_release);
    }
}

void Autolume::inferenceThreadLoop() {
    using namespace std::chrono;

    // Initialize MPS directly on this dedicated thread (like autolumelive's approach)
    try {
        std::cout << "Autolume: Initializing MPS on inference thread..." << std::endl;
        device = torch::Device(torch::kMPS);

        std::cout << "Autolume: Moving model to MPS..." << std::endl;
        model.to(device);

        std::cout << "Autolume: Moving tensor to MPS..." << std::endl;
        inputTensor = inputTensor.to(device, false, false);
        inputs.clear();
        inputs.emplace_back(inputTensor);

        mpsInitialized.store(true, std::memory_order_release);
        std::cout << "Autolume: MPS initialization complete!" << std::endl;

        // Test forward pass immediately after initialization
        try {
            std::cout << "Autolume: Testing forward pass..." << std::endl;
            torch::NoGradGuard no_grad;
            auto test_output = model.forward(inputs).toTensor();
            std::cout << "Autolume: Test forward pass succeeded! Output shape: ["
                      << test_output.size(0) << ", " << test_output.size(1) << ", "
                      << test_output.size(2) << ", " << test_output.size(3) << "]" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Autolume: Test forward pass FAILED: " << e.what() << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Autolume: MPS initialization failed: " << e.what() << std::endl;
        mpsInitialized.store(true, std::memory_order_release);
        // Exit thread if initialization failed
        return;
    }

    // Main inference loop (like autolumelive's _process_fn)
    std::cout << "Autolume: Entering inference loop..." << std::endl;
    while (!shouldExit.load(std::memory_order_acquire)) {
        // Check if inference is requested
        if (inferenceRequested.load(std::memory_order_acquire)) {
            runInference();
            inferenceRequested.store(false, std::memory_order_release);
        }

        // Sleep briefly to avoid busy-waiting
        std::this_thread::sleep_for(milliseconds(1));
    }

    std::cout << "Autolume: Inference thread exiting..." << std::endl;
}

void Autolume::requestInference() {
    // Don't request if not initialized yet
    if (!isInitialized.load(std::memory_order_acquire)) {
        return;
    }

    // Skip if inference is already running (avoid overlapping calls)
    if (inferenceRunning.load(std::memory_order_acquire)) {
        return;
    }

    // Signal inference thread to run inference (no MessageManager needed)
    inferenceRequested.store(true, std::memory_order_release);
}

void Autolume::runInference() {
    // Don't run inference if not initialized yet
    if (!isInitialized.load(std::memory_order_acquire)) {
        return;
    }

    // Mark inference as running
    inferenceRunning.store(true, std::memory_order_release);

    try {
        // Copy input if available (lock-free read from audio thread)
        std::array<float, Constants::nfft> audio_samples;
        if (inputReady.load(std::memory_order_acquire)) {
            std::copy(ordered_in_buf.begin(), ordered_in_buf.end(), audio_samples.begin());
            inputReady.store(false, std::memory_order_release);
        } else {
            // Use previous samples if no new data
            audio_samples.fill(0.0f);
        }

        // Compute FFT magnitude using Apple Accelerate vDSP
        // Step 1: Convert real input to split complex format
        // vDSP expects input as interleaved complex, reinterpret as DSPComplex
        vDSP_ctoz(reinterpret_cast<DSPComplex*>(audio_samples.data()), 2, &fftSplit, 1, Constants::nfft / 2);

        // Step 2: Perform forward FFT (real-to-complex)
        vDSP_fft_zrip(fftSetup, &fftSplit, 1, fftLog2n, FFT_FORWARD);

        vDSP_zvabs(&fftSplit, 1, inference_input_buf.data(), 1, Constants::nfft / 2);

        // Step 5: Fill second half with zeros (or mirror if you want full spectrum)
        std::fill(inference_input_buf.begin() + Constants::nfft / 2, inference_input_buf.end(), 0.0f);

        // Copy CPU buffer to MPS tensor (can't use accessor on MPS tensor)
        // Create CPU tensor from buffer, then copy to MPS
        auto cpu_tensor = torch::from_blob(
            inference_input_buf.data(),
            {1, Constants::nfft},
            torch::kFloat32
        ).clone();  // Clone to own the data

        // Copy to preallocated MPS tensor
        inputTensor.copy_(cpu_tensor);

        // Run model inference
        torch::NoGradGuard no_grad;
        auto output = model.forward(inputs).toTensor();

        // Convert output tensor to RGB image
        // Assuming output is [1, 3, 512, 512] in range [-1, 1] or [0, 1]
        output = output.squeeze(0);  // [3, 512, 512]
        output = output.permute({1, 2, 0});  // [512, 512, 3]
        output = output.to(torch::kCPU);

        // Clamp and convert to uint8
        output = (output + 1.0f) * 127.5f;  // Convert from [-1, 1] to [0, 255]
        output = output.clamp(0.0f, 255.0f);

        auto data = output.contiguous();
        auto* ptr = data.data_ptr<float>();

        // Write to the write buffer
        auto& writeBuffer = frameBuffer[writeFrameIndex];
        for (size_t i = 0; i < Constants::frameBytes; i++) {
            writeBuffer[i] = static_cast<uint8_t>(ptr[i]);
        }

        // Swap buffers atomically
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            int oldReadable = readableFrameIndex.load(std::memory_order_relaxed);
            readableFrameIndex.store(writeFrameIndex, std::memory_order_release);
            writeFrameIndex = oldReadable;
        }

        // Mark inference as complete
        inferenceRunning.store(false, std::memory_order_release);
    }
    catch (const std::exception& e) {
        std::cerr << "Autolume: Inference error: " << e.what() << std::endl;
        // Mark inference as complete even on error
        inferenceRunning.store(false, std::memory_order_release);
    }
}

bool Autolume::getLatestFrame(uint8_t* dest, size_t numBytes) {
    // Don't access frame buffers until initialization is complete
    if (!isInitialized.load(std::memory_order_acquire)) {
        return false;
    }

    if (numBytes < Constants::frameBytes) {
        return false;
    }

    // Read from the readable buffer
    int readIdx;
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        readIdx = readableFrameIndex.load(std::memory_order_acquire);
    }

    // Copy frame data
    std::copy(frameBuffer[readIdx].begin(), frameBuffer[readIdx].end(), dest);
    return true;
}

void Autolume::findNoiseStrengthParameters() {
    // Find all parameters with "noise_strength" in their name
    noiseStrengthParams.clear();

    for (const auto& param : model.named_parameters()) {
        if (param.name.find("noise_strength") != std::string::npos) {
            noiseStrengthParams.push_back(param.value);
            std::cout << "Autolume: Found noise_strength parameter: " << param.name << std::endl;
        }
    }

    std::cout << "Autolume: Cached " << noiseStrengthParams.size()
              << " noise_strength parameters" << std::endl;
}

void Autolume::setNoiseStrength(float value) {
    // Set all noise_strength parameters to the given value
    // Thread-safe: PyTorch tensor operations are thread-safe
    torch::NoGradGuard no_grad;

    for (auto& param : noiseStrengthParams) {
        param.fill_(value);
    }
}

float Autolume::getNoiseStrength() const {
    // Return the value of the first noise_strength parameter
    if (noiseStrengthParams.empty()) {
        return 0.0f;
    }

    return noiseStrengthParams[0].item<float>();
}
