#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#define CPPHTTPLIB_USE_POLL
#include "httplib.h"
#include "json.hpp"
#include "piper.hpp"
#include "wavfile.hpp"

using namespace std;
using json = nlohmann::json;

enum OutputType { OUTPUT_FILE, OUTPUT_DIRECTORY, OUTPUT_STDOUT, OUTPUT_RAW };

struct RunConfig {
  // Path to .onnx voice file
  filesystem::path modelPath;

  // Path to JSON voice config file
  filesystem::path modelConfigPath;

  // Type of output to produce.
  // Default is to write a WAV file in the current directory.
  OutputType outputType = OUTPUT_DIRECTORY;

  // Path for output
  optional<filesystem::path> outputPath = filesystem::path(".");

  // Numerical id of the default speaker (multi-speaker voices)
  optional<piper::SpeakerId> speakerId;

  // Amount of noise to add during audio generation
  optional<float> noiseScale;

  // Speed of speaking (1 = normal, < 1 is faster, > 1 is slower)
  optional<float> lengthScale;

  // Variation in phoneme lengths
  optional<float> noiseW;

  // Seconds of silence to add after each sentence
  optional<float> sentenceSilenceSeconds;

  // Path to espeak-ng data directory (default is next to piper executable)
  optional<filesystem::path> eSpeakDataPath;

  // Path to libtashkeel ort model
  // https://github.com/mush42/libtashkeel/
  optional<filesystem::path> tashkeelModelPath;

  // stdin input is lines of JSON instead of text with format:
  // {
  //   "text": str,               (required)
  //   "speaker_id": int,         (optional)
  //   "speaker": str,            (optional)
  //   "output_file": str,        (optional)
  // }
  bool jsonInput = false;

  // Seconds of extra silence to insert after a single phoneme
  optional<std::map<piper::Phoneme, float>> phonemeSilenceSeconds;

  // true to use CUDA execution provider
  bool useCuda = false;

  // true to start as http server
  bool server = false;

  // HTTP server address
  std::string address = "0.0.0.0";

  // HTTP server port
  int port = 8080;
};

void parseArgs(int argc, char *argv[], RunConfig &runConfig);
void rawOutputProc(vector<float_t>& sharedAudioBuffer, mutex& mutAudio,
                   condition_variable &cvAudio, bool &audioReady,
                   bool &audioFinished);
void textToWavFile(piper::PiperConfig& config, piper::Voice& voice, std::string text,
                   std::ostream& audioFile, piper::SynthesisResult& result);
void runCommandLine(RunConfig& runConfig, piper::PiperConfig& piperConfig, piper::Voice& voice);
void runServer(RunConfig& runConfig, piper::PiperConfig& piperConfig, piper::Voice& voice);

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  spdlog::set_default_logger(spdlog::stderr_color_st("piper"));

  RunConfig runConfig;
  parseArgs(argc, argv, runConfig);

#ifdef _WIN32
  // Required on Windows to show IPA symbols
  SetConsoleOutputCP(CP_UTF8);
#endif

  piper::PiperConfig piperConfig;
  piper::Voice voice;

  spdlog::debug("Loading voice from {} (config={})",
                runConfig.modelPath.string(),
                runConfig.modelConfigPath.string());

  auto startTime = chrono::steady_clock::now();
  loadVoice(piperConfig, runConfig.modelPath.string(),
            runConfig.modelConfigPath.string(), voice, runConfig.speakerId,
            runConfig.useCuda);
  auto endTime = chrono::steady_clock::now();
  spdlog::info("Loaded voice in {} second(s)",
               chrono::duration<double>(endTime - startTime).count());

  // Get the path to the piper executable so we can locate espeak-ng-data, etc.
  // next to it.
#ifdef _MSC_VER
  auto exePath = []() {
    wchar_t moduleFileName[MAX_PATH] = {0};
    GetModuleFileNameW(nullptr, moduleFileName, std::size(moduleFileName));
    return filesystem::path(moduleFileName);
  }();
#else
#ifdef __APPLE__
  auto exePath = []() {
    char moduleFileName[PATH_MAX] = {0};
    uint32_t moduleFileNameSize = std::size(moduleFileName);
    _NSGetExecutablePath(moduleFileName, &moduleFileNameSize);
    return filesystem::path(moduleFileName);
  }();
#else
  auto exePath = filesystem::canonical("/proc/self/exe");
#endif
#endif

  if (voice.phonemizeConfig.phonemeType == piper::eSpeakPhonemes) {
    spdlog::debug("Voice uses eSpeak phonemes ({})",
                  voice.phonemizeConfig.eSpeak.voice);

    if (runConfig.eSpeakDataPath) {
      // User provided path
      piperConfig.eSpeakDataPath = runConfig.eSpeakDataPath.value().string();
    } else {
      // Assume next to piper executable
      piperConfig.eSpeakDataPath =
          std::filesystem::absolute(
              exePath.parent_path().append("espeak-ng-data"))
              .string();

      spdlog::debug("espeak-ng-data directory is expected at {}",
                    piperConfig.eSpeakDataPath);
    }
  } else {
    // Not using eSpeak
    piperConfig.useESpeak = false;
  }

  // Enable libtashkeel for Arabic
  if (voice.phonemizeConfig.eSpeak.voice == "ar") {
    piperConfig.useTashkeel = true;
    if (runConfig.tashkeelModelPath) {
      // User provided path
      piperConfig.tashkeelModelPath =
          runConfig.tashkeelModelPath.value().string();
    } else {
      // Assume next to piper executable
      piperConfig.tashkeelModelPath =
          std::filesystem::absolute(
              exePath.parent_path().append("libtashkeel_model.ort"))
              .string();

      spdlog::debug("libtashkeel model is expected at {}",
                    piperConfig.tashkeelModelPath.value());
    }
  }

  piper::initialize(piperConfig, voice);

  // Scales
  if (runConfig.noiseScale) {
    voice.synthesisConfig.noiseScale = runConfig.noiseScale.value();
  }

  if (runConfig.lengthScale) {
    voice.synthesisConfig.lengthScale = runConfig.lengthScale.value();
  }

  if (runConfig.noiseW) {
    voice.synthesisConfig.noiseW = runConfig.noiseW.value();
  }

  if (runConfig.sentenceSilenceSeconds) {
    voice.synthesisConfig.sentenceSilenceSeconds =
        runConfig.sentenceSilenceSeconds.value();
  }

  if (runConfig.phonemeSilenceSeconds) {
    if (!voice.synthesisConfig.phonemeSilenceSeconds) {
      // Overwrite
      voice.synthesisConfig.phonemeSilenceSeconds =
          runConfig.phonemeSilenceSeconds;
    } else {
      // Merge
      for (const auto &[phoneme, silenceSeconds] :
           *runConfig.phonemeSilenceSeconds) {
        voice.synthesisConfig.phonemeSilenceSeconds->try_emplace(
            phoneme, silenceSeconds);
      }
    }

  } // if phonemeSilenceSeconds

  if (runConfig.server) {
    runServer(runConfig, piperConfig, voice);
  } else {
    runCommandLine(runConfig, piperConfig, voice);
  }

  piper::terminate(piperConfig);

  return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------

void runServer(RunConfig& runConfig, piper::PiperConfig& piperConfig, piper::Voice& voice) {
  using namespace httplib;

  Server svr;

  // FIXME make it work
  // svr.new_task_queue = [] { return new ThreadPool(
  //   /*num_threads=*/CPPHTTPLIB_THREAD_POOL_COUNT,
  //   /*max_queued_requests=*/36);
  // };

  svr.set_exception_handler([](const Request& req, Response& res, std::exception_ptr ep) {
    res.status = StatusCode::InternalServerError_500;
    try {
      std::rethrow_exception(ep);
    } catch (std::exception& e) {
      spdlog::error("unhandled {} -> {}\n{}", req.body, res.status, e.what());
    } catch (...) {
      // if you don't provide the catch (...) block for a rethrown exception pointer, 
      // an uncaught exception will end up causing the server crash.
      // Be careful!
      spdlog::error("unhandled {} -> {}\n{}", req.body, res.status, "Unknown Exception");
    }
  });

  svr.set_logger([](const Request& req, const Response& res) {
    spdlog::debug("handled {} -> {}", req.body, res.status);
  });

  svr.Post("/tts", [&piperConfig, &voice](const Request& req, Response& res) {
    json data = json::parse(req.body);
    std::string line = data["text"];

    res.set_chunked_content_provider("audio/wav", [&, line](size_t offset, DataSink& sink) -> bool {
      piper::SynthesisResult result;
      result.startTime = std::chrono::steady_clock::now();

      WavHeader header;
      fillWavHeader(header, voice.synthesisConfig.sampleRate, voice.synthesisConfig.sampleWidth, voice.synthesisConfig.channels, -1);
      sink.write((char*)(&header), sizeof(header));

      piper::textToAudio(piperConfig, voice, line, result, [&sink](std::vector<float_t> const& pcm32Audio) {
        vector<int16_t> audioBuffer;
        piper::pcm32_to_pcm16(pcm32Audio, audioBuffer);
        sink.write((char*)(audioBuffer.data()), sizeof(int16_t) / sizeof(char) * audioBuffer.size());
      });

      sink.done();

      auto endTime = std::chrono::steady_clock::now();
      auto durationSeconds = std::chrono::duration<double>(endTime - result.startTime).count();
      spdlog::info("Real-time factor: {} (audio={} sec, infer={} sec, total={} sec)",
        durationSeconds / result.audioSeconds,
        result.audioSeconds,
        result.inferSeconds,
        durationSeconds
      );

      return true;
    });
  });

  spdlog::info("Starting server http://{}:{}", runConfig.address, runConfig.port);
  svr.listen(runConfig.address, runConfig.port);
}

void runCommandLine(RunConfig& runConfig, piper::PiperConfig& piperConfig, piper::Voice& voice) {
  if (runConfig.outputType == OUTPUT_DIRECTORY) {
    runConfig.outputPath = filesystem::absolute(runConfig.outputPath.value());
    spdlog::info("Output directory: {}", runConfig.outputPath.value().string());
  }

  string line;
  piper::SynthesisResult result;
  result.startTime = std::chrono::steady_clock::now();
  while (getline(cin, line)) {
    auto outputType = runConfig.outputType;
    auto speakerId = voice.synthesisConfig.speakerId;
    std::optional<filesystem::path> maybeOutputPath = runConfig.outputPath;

    if (runConfig.jsonInput) {
      // Each line is a JSON object
      json lineRoot = json::parse(line);

      // Text is required
      line = lineRoot["text"].get<std::string>();

      if (lineRoot.contains("output_file")) {
        // Override output WAV file path
        outputType = OUTPUT_FILE;
        maybeOutputPath =
          filesystem::path(lineRoot["output_file"].get<std::string>());
      }

      if (lineRoot.contains("speaker_id")) {
        // Override speaker id
        voice.synthesisConfig.speakerId =
          lineRoot["speaker_id"].get<piper::SpeakerId>();
      } else if (lineRoot.contains("speaker")) {
        // Resolve to id using speaker id map
        auto speakerName = lineRoot["speaker"].get<std::string>();
        if ((voice.modelConfig.speakerIdMap) &&
          (voice.modelConfig.speakerIdMap->count(speakerName) > 0)) {
          voice.synthesisConfig.speakerId =
            (*voice.modelConfig.speakerIdMap)[speakerName];
        } else {
          spdlog::warn("No speaker named: {}", speakerName);
        }
      }
    }

    if (outputType == OUTPUT_DIRECTORY) {
      // Timestamp is used for path to output WAV file
      const auto now = chrono::system_clock::now();
      const auto timestamp = chrono::duration_cast<chrono::nanoseconds>(now.time_since_epoch()).count();
      const auto t = std::chrono::system_clock::to_time_t(now);

      // Generate path using timestamp in a human readable format close to ISO8601
      stringstream outputName;
      outputName << std::put_time(std::localtime(&t), "%FT%T.") << (timestamp % 1000000000) / 1000 << ".wav";
      filesystem::path outputPath = runConfig.outputPath.value();
      outputPath.append(outputName.str());

      // Output audio to automatically-named WAV file in a directory
      ofstream audioFile(outputPath.string(), ios::binary);
      textToWavFile(piperConfig, voice, line, audioFile, result);
      cout << outputPath.string() << endl;
    } else if (outputType == OUTPUT_FILE) {
      if (!maybeOutputPath || maybeOutputPath->empty()) {
        throw runtime_error("No output path provided");
      }

      filesystem::path outputPath = maybeOutputPath.value();

      if (!runConfig.jsonInput) {
        // Read all of standard input before synthesizing.
        // Otherwise, we would overwrite the output file for each line.
        stringstream text;
        text << line;
        while (getline(cin, line)) {
          text << " " << line;
        }

        line = text.str();
      }

      // Output audio to WAV file
      ofstream audioFile(outputPath.string(), ios::binary);
      textToWavFile(piperConfig, voice, line, audioFile, result);
      cout << outputPath.string() << endl;
    } else if (outputType == OUTPUT_STDOUT) {
      // Output WAV to stdout
      textToWavFile(piperConfig, voice, line, cout, result);
    } else if (outputType == OUTPUT_RAW) {
      // Raw output to stdout
      mutex mutAudio;
      condition_variable cvAudio;
      bool audioReady = false;
      bool audioFinished = false;
      vector<float_t> sharedAudioBuffer;

#ifdef _WIN32
      // Needed on Windows to avoid terminal conversions
      setmode(fileno(stdout), O_BINARY);
      setmode(fileno(stdin), O_BINARY);
#endif

      thread rawOutputThread(rawOutputProc, ref(sharedAudioBuffer),
        ref(mutAudio), ref(cvAudio), ref(audioReady),
        ref(audioFinished));

      piper::textToAudio(piperConfig, voice, line, result, [&sharedAudioBuffer, &mutAudio,
        &cvAudio, &audioReady](std::vector<float_t> const& pcm32Audio) {
        // Signal thread that audio is ready
          {
            unique_lock lockAudio(mutAudio);
            copy(pcm32Audio.begin(), pcm32Audio.end(), back_inserter(sharedAudioBuffer));
            audioReady = true;
            cvAudio.notify_one();
          }
      });

      // Signal thread that there is no more audio
      {
        unique_lock lockAudio(mutAudio);
        audioReady = true;
        audioFinished = true;
        cvAudio.notify_one();
      }

      // Wait for audio output to finish
      spdlog::info("Waiting for audio to finish playing...");
      rawOutputThread.join();
    }

    auto endTime = std::chrono::steady_clock::now();
    auto durationSeconds = std::chrono::duration<double>(endTime - result.startTime).count();
    spdlog::info("Real-time factor: {} (audio={} sec, infer={} sec, total={} sec)",
      durationSeconds / result.audioSeconds,
      result.audioSeconds,
      result.inferSeconds,
      durationSeconds
    );

    // Restore config (--json-input)
    voice.synthesisConfig.speakerId = speakerId;

  } // for each line
}

void rawOutputProc(vector<float_t>& sharedAudioBuffer, mutex& mutAudio,
                   condition_variable &cvAudio, bool &audioReady,
                   bool &audioFinished) {
  vector<int16_t> internalAudioBuffer;
  while (true) {
    {
      unique_lock lockAudio{mutAudio};
      cvAudio.wait(lockAudio, [&audioReady] { return audioReady; });

      if (sharedAudioBuffer.empty() && audioFinished) {
        break;
      }

      piper::pcm32_to_pcm16(sharedAudioBuffer, internalAudioBuffer);

      sharedAudioBuffer.clear();

      if (!audioFinished) {
        audioReady = false;
      }
    }

    cout.write((const char *)internalAudioBuffer.data(),
               sizeof(int16_t) * internalAudioBuffer.size());
    cout.flush();
    internalAudioBuffer.clear();
  }

} // rawOutputProc

// Phonemize text and synthesize audio to WAV file
void textToWavFile(piper::PiperConfig& config, piper::Voice& voice, std::string text,
                   std::ostream& audioFile, piper::SynthesisResult& result) {

  std::vector<int16_t> audioBuffer;
  textToAudio(config, voice, text, result, [&audioBuffer](std::vector<float_t> const& pcm32Audio) {
    piper::pcm32_to_pcm16(pcm32Audio, audioBuffer);
  });

  // Write WAV
  auto synthesisConfig = voice.synthesisConfig;
  writeWavHeader(synthesisConfig.sampleRate, synthesisConfig.sampleWidth,
                 synthesisConfig.channels, (int32_t)audioBuffer.size(),
                 audioFile);

  audioFile.write((const char*)audioBuffer.data(),
                  sizeof(int16_t) * audioBuffer.size());

} /* textToWavFile */

// ----------------------------------------------------------------------------

void printUsage(char *argv[]) {
  cerr << endl;
  cerr << "usage: " << argv[0] << " [options]" << endl;
  cerr << endl;
  cerr << "options:" << endl;
  cerr << "   -h        --help              show this message and exit" << endl;
  cerr << "   -m  FILE  --model       FILE  path to onnx model file" << endl;
  cerr << "   -c  FILE  --config      FILE  path to model config file "
          "(default: model path + .json)"
       << endl;
  cerr << "   -f  FILE  --output_file FILE  path to output WAV file ('-' for "
          "stdout)"
       << endl;
  cerr << "   -d  DIR   --output_dir  DIR   path to output directory (default: "
          "cwd)"
       << endl;
  cerr << "   --output_raw                  output raw audio to stdout as it "
          "becomes available"
       << endl;
  cerr << "   -s  NUM   --speaker     NUM   id of speaker (default: 0)" << endl;
  cerr << "   --noise_scale           NUM   generator noise (default: 0.667)"
       << endl;
  cerr << "   --length_scale          NUM   phoneme length (default: 1.0)"
       << endl;
  cerr << "   --noise_w               NUM   phoneme width noise (default: 0.8)"
       << endl;
  cerr << "   --sentence_silence      NUM   seconds of silence after each "
          "sentence (default: 0.2)"
       << endl;
  cerr << "   --espeak_data           DIR   path to espeak-ng data directory"
       << endl;
  cerr << "   --tashkeel_model        FILE  path to libtashkeel onnx model "
          "(arabic)"
       << endl;
  cerr << "   --json-input                  stdin input is lines of JSON "
          "instead of plain text"
       << endl;
  cerr << "   --use-cuda                    use CUDA execution provider"
       << endl;
  cerr << "   --server                      (experimental) start as HTTP server"
    << endl;
  cerr << "   --address                     (experimental) start as HTTP server on address"
    << endl;
  cerr << "   --port                        (experimental) start as HTTP server on port"
    << endl;
  cerr << "   --debug                       print DEBUG messages to the console"
       << endl;
  cerr << "   -q       --quiet              disable logging" << endl;
  cerr << endl;
}

void ensureArg(int argc, char *argv[], int argi) {
  if ((argi + 1) >= argc) {
    printUsage(argv);
    exit(0);
  }
}

// Parse command-line arguments
void parseArgs(int argc, char *argv[], RunConfig &runConfig) {
  optional<filesystem::path> modelConfigPath;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-m" || arg == "--model") {
      ensureArg(argc, argv, i);
      runConfig.modelPath = filesystem::path(argv[++i]);
    } else if (arg == "-c" || arg == "--config") {
      ensureArg(argc, argv, i);
      modelConfigPath = filesystem::path(argv[++i]);
    } else if (arg == "-f" || arg == "--output_file" ||
               arg == "--output-file") {
      ensureArg(argc, argv, i);
      std::string filePath = argv[++i];
      if (filePath == "-") {
        runConfig.outputType = OUTPUT_STDOUT;
        runConfig.outputPath = nullopt;
      } else {
        runConfig.outputType = OUTPUT_FILE;
        runConfig.outputPath = filesystem::path(filePath);
      }
    } else if (arg == "-d" || arg == "--output_dir" || arg == "output-dir") {
      ensureArg(argc, argv, i);
      runConfig.outputType = OUTPUT_DIRECTORY;
      runConfig.outputPath = filesystem::path(argv[++i]);
    } else if (arg == "--output_raw" || arg == "--output-raw") {
      runConfig.outputType = OUTPUT_RAW;
    } else if (arg == "-s" || arg == "--speaker") {
      ensureArg(argc, argv, i);
      runConfig.speakerId = (piper::SpeakerId)stol(argv[++i]);
    } else if (arg == "--noise_scale" || arg == "--noise-scale") {
      ensureArg(argc, argv, i);
      runConfig.noiseScale = stof(argv[++i]);
    } else if (arg == "--length_scale" || arg == "--length-scale") {
      ensureArg(argc, argv, i);
      runConfig.lengthScale = stof(argv[++i]);
    } else if (arg == "--noise_w" || arg == "--noise-w") {
      ensureArg(argc, argv, i);
      runConfig.noiseW = stof(argv[++i]);
    } else if (arg == "--sentence_silence" || arg == "--sentence-silence") {
      ensureArg(argc, argv, i);
      runConfig.sentenceSilenceSeconds = stof(argv[++i]);
    } else if (arg == "--phoneme_silence" || arg == "--phoneme-silence") {
      ensureArg(argc, argv, i);
      ensureArg(argc, argv, i + 1);
      auto phonemeStr = std::string(argv[++i]);
      if (!piper::isSingleCodepoint(phonemeStr)) {
        std::cerr << "Phoneme '" << phonemeStr
          << "' is not a single codepoint (--phoneme_silence)"
          << std::endl;
        exit(1);
      }

      if (!runConfig.phonemeSilenceSeconds) {
        runConfig.phonemeSilenceSeconds.emplace();
      }

      auto phoneme = piper::getCodepoint(phonemeStr);
      (*runConfig.phonemeSilenceSeconds)[phoneme] = stof(argv[++i]);
    } else if (arg == "--espeak_data" || arg == "--espeak-data") {
      ensureArg(argc, argv, i);
      runConfig.eSpeakDataPath = filesystem::path(argv[++i]);
    } else if (arg == "--tashkeel_model" || arg == "--tashkeel-model") {
      ensureArg(argc, argv, i);
      runConfig.tashkeelModelPath = filesystem::path(argv[++i]);
    } else if (arg == "--json_input" || arg == "--json-input") {
      runConfig.jsonInput = true;
    } else if (arg == "--use_cuda" || arg == "--use-cuda") {
      runConfig.useCuda = true;
    } else if (arg == "--server") {
      runConfig.server = true;
    } else if (arg == "--address") {
      ensureArg(argc, argv, i);
      runConfig.address = std::string(argv[++i]);
    } else if (arg == "--port") {
      ensureArg(argc, argv, i);
      runConfig.port = atoi(argv[++i]);
    } else if (arg == "--version") {
      std::cout << piper::getVersion() << std::endl;
      exit(0);
    } else if (arg == "--debug") {
      // Set DEBUG logging
      spdlog::set_level(spdlog::level::debug);
    } else if (arg == "-q" || arg == "--quiet") {
      // diable logging
      spdlog::set_level(spdlog::level::off);
    } else if (arg == "-h" || arg == "--help") {
      printUsage(argv);
      exit(0);
    }
  }

  // Verify model file exists
  ifstream modelFile(runConfig.modelPath.c_str(), ios::binary);
  if (!modelFile.good()) {
    throw runtime_error("Model file doesn't exist");
  }

  if (!modelConfigPath) {
    runConfig.modelConfigPath =
        filesystem::path(runConfig.modelPath.string() + ".json");
  } else {
    runConfig.modelConfigPath = modelConfigPath.value();
  }

  // Verify model config exists
  ifstream modelConfigFile(runConfig.modelConfigPath.c_str());
  if (!modelConfigFile.good()) {
    throw runtime_error("Model config doesn't exist");
  }
}
