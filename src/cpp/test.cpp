#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "piper.hpp"

using namespace std;
using json = nlohmann::json;

int main(int argc, char *argv[]) {
  piper::PiperConfig piperConfig;
  piper::Voice voice;

  if (argc < 2) {
    std::cerr << "Need voice model path" << std::endl;
    return 1;
  }

  if (argc < 3) {
    std::cerr << "Need espeak-ng-data path" << std::endl;
    return 1;
  }

  if (argc < 4) {
    std::cerr << "Need output WAV path" << std::endl;
    return 1;
  }

  auto modelPath = std::string(argv[1]);
  piperConfig.eSpeakDataPath = std::string(argv[2]);
  auto outputPath = std::string(argv[3]);

  optional<piper::SpeakerId> speakerId;
  loadVoice(piperConfig, modelPath, modelPath + ".json", voice, speakerId,
            false);
  piper::initialize(piperConfig, voice);

  piper::SynthesisResult result;

  std::vector<float_t> audioBuffer;
  piper::textToAudio(piperConfig, voice, "This is a test.", result, [&audioBuffer](std::vector<float_t> const& pcm32Audio) {
    std::copy(pcm32Audio.cbegin(), pcm32Audio.cend(), std::back_inserter(audioBuffer));
  });

  piper::terminate(piperConfig);

  // Verify that file has some data
  if (audioBuffer.size() < 10000) {
    std::cerr << "ERROR: Output is smaller than expected!" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "OK" << std::endl;

  return EXIT_SUCCESS;
}
