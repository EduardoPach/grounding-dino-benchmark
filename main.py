import argparse

import requests
from PIL import Image
from transformers import GroundingDinoForObjectDetection, AutoProcessor

from benchmarker import Benchmarker

def prepare_inputs():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image, "a cat. a remote control."

def create_parser() -> argparse.Namespace:
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some arguments.')

    # Add arguments
    parser.add_argument(
        '--max-batch-size', 
        type=int, 
        default=32, 
        help='Set the maximum batch size (default: 32)'
    )
    parser.add_argument(
        '--compile', 
        action='store_true', 
        help='Enable compile mode'
    )
    parser.add_argument(
        '--reduce-overhead', 
        action='store_true', 
        help='Enable overhead reduction'
    )
    parser.add_argument(
        '--set-high-precision', 
        action='store_true', 
        help='Enable high precision mode'
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Set the number of times the benchmark for a specific batch size should run"
    )

    return parser

def main(args: argparse.Namespace) -> None:
    max_batch_size = args.max_batch_size
    num_runs = args.num_runs
    compile = args.compile
    reduce_overhead = args.reduce_overhead
    set_high_precision = args.set_high_precision

    model_id = "IDEA-Research/grounding-dino-tiny"

    model = GroundingDinoForObjectDetection.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    image, text = prepare_inputs()
    inputs = processor(images=image, text=text, return_tensors="pt")

    print("Initializing benchmarker")
    benchmark = Benchmarker(model, inputs)

    print("starting runs")
    for batch_size in range(1, max_batch_size + 1):
        # No optimization
        benchmark.run(
            batch_size=batch_size,
            compile=compile,
            set_high_precision=set_high_precision,
            reduce_overhead=reduce_overhead,
            num_runs=num_runs,
        )

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)