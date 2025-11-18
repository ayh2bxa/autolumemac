#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    juce::ignoreUnused (processorRef);
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (Constants::frameWidth*2, Constants::frameHeight);
    startTimerHz(Constants::fps);

    // Setup noise strength slider
    noiseSlider.setSliderStyle(juce::Slider::LinearVertical);
    noiseSlider.setRange(0.0, 1.0, 0.01);
    noiseSlider.setValue(0.0);
    noiseSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 80, 20);
    noiseSlider.onValueChange = [this]() {
        processorRef.renderer.setNoiseStrength(static_cast<float>(noiseSlider.getValue()));
    };
    addAndMakeVisible(noiseSlider);

    noiseLabel.setText("Noise Strength", juce::dontSendNotification);
    noiseLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(noiseLabel);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    // Left half: video rendering
    auto leftHalf = juce::Rectangle<float>(0, 0, (float) Constants::frameWidth, (float) Constants::frameHeight);
    g.drawImage(image, leftHalf);

    // Right half: GUI controls (blank for now)
    auto rightHalf = juce::Rectangle<float>((float) Constants::frameWidth, 0,
                                           (float) Constants::frameWidth, (float) Constants::frameHeight);
    g.setColour(juce::Colour(0xff2a2a2a));  // Dark grey background
    g.fillRect(rightHalf);
}

void AudioPluginAudioProcessorEditor::resized()
{
    // Right half of the screen
    auto rightHalf = getLocalBounds().removeFromRight(Constants::frameWidth);

    // Center the slider vertically in the right half, with some margin
    int sliderWidth = 60;
    int sliderHeight = 300;
    int margin = 20;

    auto sliderArea = rightHalf.withSizeKeepingCentre(sliderWidth, sliderHeight);
    noiseSlider.setBounds(sliderArea);

    // Label above the slider
    auto labelArea = sliderArea.removeFromTop(30).translated(0, -40);
    noiseLabel.setBounds(labelArea);
}

void AudioPluginAudioProcessorEditor::timerCallback()
{
    // Safety check: don't access renderer until it's initialized
    // This prevents crashes during early initialization
    if (!processorRef.renderer.isReady()) {
        return;
    }

    // Request new inference (will be skipped if already running)
    processorRef.renderer.requestInference();

    if (processorRef.renderer.getLatestFrame(frameData.data(), frameData.size())) {
        // Convert RGB data to JUCE Image
        image = juce::Image(juce::Image::RGB, Constants::frameWidth, Constants::frameHeight, false);

        juce::Image::BitmapData bitmap(image, juce::Image::BitmapData::writeOnly);
        for (int y = 0; y < Constants::frameHeight; y++) {
            for (int x = 0; x < Constants::frameWidth; x++) {
                size_t idx = (y * Constants::frameWidth + x) * 3;
                uint8_t r = frameData[idx + 0];
                uint8_t g = frameData[idx + 1];
                uint8_t b = frameData[idx + 2];
                bitmap.setPixelColour(x, y, juce::Colour(r, g, b));
            }
        }

        // Trigger repaint
        repaint();
    }
}
