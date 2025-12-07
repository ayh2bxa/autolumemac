#pragma once

#include "PluginProcessor.h"
#include "defines.h"

//==============================================================================
using namespace std;
class AudioPluginAudioProcessorEditor final : public juce::AudioProcessorEditor, public juce::Timer
{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

    void timerCallback() override;
private:
    bool timerStarted = false;
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AudioPluginAudioProcessor& processorRef;
    juce::Image image;
    array<uint8_t, Constants::frameBytes> frameData;

    // Model loading
    juce::TextButton uploadButton;
    juce::Label modelPathLabel;

    // Noise strength control
    juce::Slider noiseSlider;
    juce::Label noiseLabel;

    // Latent speed control
    juce::Slider speedSlider;
    juce::Label speedLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};
