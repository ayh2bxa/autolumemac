cmake_minimum_required(VERSION 3.22)

# Find LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set platform-specific plugin formats and copy directories
if(APPLE)
    set(PLUGIN_FORMATS AU VST3 Standalone)
    set(VST3_COPY_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/VST3")
    set(AU_COPY_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/Components")
elseif(WIN32)
    set(PLUGIN_FORMATS VST3 Standalone)
    set(VST3_COPY_DIR "$ENV{COMMONPROGRAMFILES}/VST3")
else() # Linux
    set(PLUGIN_FORMATS VST3 Standalone)
    set(VST3_COPY_DIR "$ENV{HOME}/.vst3")
endif()

juce_add_plugin(${PROJECT_NAME}
    COMPANY_NAME "projectfmusic"
    FORMATS ${PLUGIN_FORMATS}
    PRODUCT_NAME "AutolumeJUCE"
    MICROPHONE_PERMISSION_ENABLED TRUE
    COPY_PLUGIN_AFTER_BUILD TRUE
    VST3_COPY_DIR ${VST3_COPY_DIR}
    $<$<PLATFORM_ID:Darwin>:AU_COPY_DIR ${AU_COPY_DIR}>
)

file(GLOB_RECURSE SRC "source/*.cpp")
file(GLOB_RECURSE HDR "include/*.h")

target_sources(${PROJECT_NAME}
    PRIVATE
        ${SRC}
        ${HDR}
)
source_group("Header Files" FILES ${HDR})
source_group("Source Files" FILES ${SRC})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        include
        ${JUCE_DIR}/modules
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
        ${TORCH_LIBRARIES}
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Link macOS-specific frameworks
if(APPLE)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            "-framework Accelerate"
            "-framework SystemConfiguration"
            "-framework CoreFoundation"
    )
endif()

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JucePlugin_Enable_IAA=1
        JucePlugin_StandaloneEnableAudioInput=1
)