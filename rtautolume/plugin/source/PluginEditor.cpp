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

    // Setup upload button
    uploadButton.setButtonText("Load Model...");
    uploadButton.onClick = [this]() {
        auto fileChooser = std::make_shared<juce::FileChooser>(
            "Select TorchScript Model (.pt)",
            juce::File(),
            "*.pt"
        );

        auto chooserFlags = juce::FileBrowserComponent::openMode |
                           juce::FileBrowserComponent::canSelectFiles;

        fileChooser->launchAsync(chooserFlags, [this, fileChooser](const juce::FileChooser& fc) {
            auto file = fc.getResult();
            if (file != juce::File{}) {
                std::string path = file.getFullPathName().toStdString();
                bool success = processorRef.renderer.loadModel(path);

                if (success) {
                    modelPathLabel.setText(file.getFileName(), juce::dontSendNotification);
                    noiseSlider.setEnabled(true);
                    speedSlider.setEnabled(true);
                } else {
                    modelPathLabel.setText("Failed to load model", juce::dontSendNotification);
                    noiseSlider.setEnabled(false);
                    speedSlider.setEnabled(false);
                }
            }
        });
    };
    addAndMakeVisible(uploadButton);

    // Setup model path label
    modelPathLabel.setText("No model loaded", juce::dontSendNotification);
    modelPathLabel.setJustificationType(juce::Justification::centred);
    modelPathLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(modelPathLabel);

    // Setup noise strength slider
    noiseSlider.setSliderStyle(juce::Slider::LinearVertical);
    noiseSlider.setRange(0.0, 1.0, 0.01);
    noiseSlider.setValue(0.0);
    noiseSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 80, 20);
    noiseSlider.onValueChange = [this]() {
        processorRef.renderer.setNoiseStrength(static_cast<float>(noiseSlider.getValue()/10.f));
    };
    noiseSlider.setEnabled(false);  // Disabled until model is loaded
    addAndMakeVisible(noiseSlider);

    noiseLabel.setText("Noise Strength", juce::dontSendNotification);
    noiseLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(noiseLabel);

    // Setup latent speed slider
    speedSlider.setSliderStyle(juce::Slider::LinearVertical);
    speedSlider.setRange(-5.0, 5.0, 0.01);
    speedSlider.setValue(0.25);
    speedSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 80, 20);
    speedSlider.setSkewFactorFromMidPoint(1.0);
    speedSlider.onValueChange = [this]() {
        processorRef.renderer.setLatentSpeed(static_cast<float>(speedSlider.getValue()));
    };
    speedSlider.setEnabled(false);
    addAndMakeVisible(speedSlider);

    speedLabel.setText("Latent Speed", juce::dontSendNotification);
    speedLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(speedLabel);
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

    int margin = 20;

    // Upload button at the top
    auto topArea = rightHalf.removeFromTop(80).reduced(margin);
    auto buttonArea = topArea.removeFromTop(40);
    uploadButton.setBounds(buttonArea);

    // Model path label below button
    topArea.removeFromTop(5);  // Small gap
    auto modelLabelArea = topArea.removeFromTop(30);
    modelPathLabel.setBounds(modelLabelArea);

    // Position sliders side by side
    int sliderWidth = 60;
    int sliderHeight = 300;
    int sliderSpacing = 40;

    // Noise slider on the left
    auto noiseSliderArea = rightHalf.withSizeKeepingCentre(sliderWidth, sliderHeight)
                                    .translated(-sliderWidth/2 - sliderSpacing/2, 0);
    noiseSlider.setBounds(noiseSliderArea);

    int labelWidth = 150;
    auto noiseLabelArea = rightHalf.withSizeKeepingCentre(labelWidth, 30)
                                   .translated(-sliderWidth/2 - sliderSpacing/2, -sliderHeight/2 - 50);
    noiseLabel.setBounds(noiseLabelArea);

    // Speed slider on the right
    auto speedSliderArea = rightHalf.withSizeKeepingCentre(sliderWidth, sliderHeight)
                                    .translated(sliderWidth/2 + sliderSpacing/2, 0);
    speedSlider.setBounds(speedSliderArea);

    auto speedLabelArea = rightHalf.withSizeKeepingCentre(labelWidth, 30)
                                   .translated(sliderWidth/2 + sliderSpacing/2, -sliderHeight/2 - 50);
    speedLabel.setBounds(speedLabelArea);
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
