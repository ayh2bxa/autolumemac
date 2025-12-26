#pragma once

#include "defines.h"
#include <cstring>

class AudioFX
{
public:
    AudioFX() : sampleRate(44100.0), isInitialized(false) {}

    virtual ~AudioFX() = default;

    virtual void initialize(double sampleRate) {
        this->sampleRate = sampleRate;
        this->isInitialized = true;
        onSampleRateChanged();
    }

    virtual void reset() {
        isInitialized = false;
    }

    void setSampleRate(double newSampleRate) {
        if (sampleRate != newSampleRate) {
            sampleRate = newSampleRate;
            onSampleRateChanged();
        }
    }

    double getSampleRate() const {
        return sampleRate;
    }

    bool getIsInitialized() const {
        return isInitialized;
    }

    virtual void apply(float *in, float *out, int numSamples) = 0;

protected:
    virtual void onSampleRateChanged() {}

    double sampleRate;
    bool isInitialized;
};
