import logging
import os
from pathlib import Path
import subprocess

from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
import onnx, json
import numpy as np

phonemizer_path = Path(
    "/Users/serhiy.masyutin/projects/personal/piper-tts/piper-phonemize/"
)
phonemizer_cmd = [
    phonemizer_path / "build/piper_phonemize",
    "-l",
    "en-us",
    "--espeak-data",
    phonemizer_path / "install/share/espeak-ng-data",
]

def phonemize(text: str) -> list[int]:
    phonemized, _ = subprocess.Popen(phonemizer_cmd, 
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE
                        ).communicate(input=text.encode())

    parsed = json.loads(phonemized)
    return parsed["phoneme_ids"]

class PiperCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_config_path: Path):
        with open(model_config_path) as f:
            config = json.load(f)

        self._scales = np.array(
            [
                config["inference"]["noise_scale"],
                config["inference"]["length_scale"],
                config["inference"]["noise_w"],
            ],
            dtype=np.float32,
        )

        num_speakers = config["num_speakers"]
        if num_speakers <= 1:
            speaker_id = None

        if (num_speakers > 1) and (speaker_id is None):
            # Default speaker
            speaker_id = 0

        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)
            self._sid = sid

        sample_texts = ["what is that? Hooray! Hooray", "This is another test!", "This is a test."]
        self._samples = iter([phonemize(t) for t in sample_texts])


    def get_next(self) -> dict:
        phoneme_ids = next(self._samples, None)
        if phoneme_ids:
            phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
            phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)

            return {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_ids_lengths,
                "scales": self._scales,
            }

        return None


def quantize_onnx_model(onnx_model_path, preprocessed_model_path, quantized_model_path):
    onnx.load(onnx_model_path)

    quant_pre_process(
        input_model_path=onnx_model_path, output_model_path=preprocessed_model_path
    )

    onnx.load(preprocessed_model_path)
    quantize_static(
        preprocessed_model_path,
        quantized_model_path,
        calibration_data_reader=PiperCalibrationDataReader(
            model_config_path=input.with_suffix(input.suffix + ".json")
        ),
        weight_type=QuantType.QInt16,
        activation_type=QuantType.QInt16,
        calibrate_method=CalibrationMethod.MinMax
    )

    onnx.load(quantized_model_path)

    logging.info(f"quantized model saved to:{quantized_model_path}")

input = Path("./src/python/python_optimize/lessac/high/en_US-lessac-high.onnx")
preprocessed = Path(
    "./src/python/python_optimize/lessac/high/en_US-lessac-high.pre.onnx"
)
output = Path("./src/python/python_optimize/lessac/high/en_US-lessac-high.quant.onnx")

quantize_onnx_model(input, preprocessed, output)

print("ONNX full precision model size (MB):", os.path.getsize(input) / (1024 * 1024))
print("ONNX quantized model size (MB):", os.path.getsize(output) / (1024 * 1024))
