#pragma once

#include <stddef.h>

namespace Constants {
    static constexpr int max_buf_size = 8192;
    static constexpr int nfft = 512;
    static constexpr int frameWidth  = 512;
    static constexpr int frameHeight = 512;
    static constexpr int frameNumCh = 3;
    static constexpr int frameBytes = frameWidth * frameHeight * frameNumCh;
    static constexpr int fps = 20;
    static constexpr double target_sr = 16000.0;
}
