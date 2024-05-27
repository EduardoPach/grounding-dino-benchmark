import os
import time
from typing import Optional, Dict, Union

import torch
import pandas as pd
from tqdm import tqdm
from transformers.tokenization_utils import BatchEncoding

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

class Benchmarker:
    def __init__(
        self, 
        model: torch.nn.Module, 
        inputs: BatchEncoding, 
        device: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> None:
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.inputs = inputs.to(self.device)
        self.output_dir = output_dir if output_dir is not None else os.path.join(os.path.dirname(__file__), "results")

    def batch_input(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batched_inputs = {}
        for key, val in self.inputs.items():
            dims = val.ndim
            new_shape = [batch_size if i == 0 else 1 for i in range(dims)]
            batched_inputs[key] = val.repeat(*new_shape)

        return batched_inputs
    
    def save_results(self, results: pd.DataFrame, filename: str) -> None:
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if os.path.exists(filepath):
            print("File already exist overriding")
        results.to_csv(
            os.path.join(self.output_dir, filename),
            index=False
        )

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"

    @property
    def context_length(self) -> Union[int, None]:
        if "input_ids" in self.inputs:
            return self.inputs.input_ids.shape[-1]
    
    @property
    def gpu_name(self) -> str:
        if self.is_gpu:
            return f"{torch.cuda.get_device_name(self.device)}"
        else:
            "CPU"
    
    def reset(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def run_once(self, model: torch.nn.Module, batched_inputs: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, float]:
        result = {}

        self.reset()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = model(**batched_inputs)
        end.record()

        torch.cuda.synchronize()

        result["peak_memory"] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2 # peak memory as MB
        result["latency"] = start.elapsed_time(end) # ms
        result["throughput"] = batch_size / (result["latency"] / 1000.0) # samples per second

        self.reset()

        return result

    def run(
        self, 
        batch_size: int,
        compile: bool, 
        set_high_precision: bool, 
        reduce_overhead: bool,
        num_runs: int = 100,
        warmup: bool = True,
        warmup_iterations: int = 1
    ) -> None:
        model = self.model

        if compile and not reduce_overhead:
            print("Compiling...")
            model = torch.compile(model)
    
        if compile and reduce_overhead:
            model = torch.compile(model, mode="reduce-overhead")
        
        if set_high_precision:
            torch.set_float32_matmul_precision("high")

        if warmup:
            for i in range(warmup_iterations):
                _ = model(**self.inputs)
        
        results = []
        batched_inputs = self.batch_input(batch_size)
        description = f"Benchmarking with batch-size: {batch_size} - compile: {compile} - reduce-overhead: {reduce_overhead} - high-precision: {set_high_precision}"
        for iteration in tqdm(range(num_runs), desc=description, unit="# Run"):
            time.sleep(1)
            try:
                result = self.run_once(model, batched_inputs, batch_size)
                results.append(result)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"Out of memory error caught at iteration {i}")
                    torch.cuda.empty_cache()
                    result = {
                        "peak_memory": torch.cuda.max_memory_allocated(self.device) / 1024 ** 2,
                        "latency": None,
                        "throughput": None,
                        "out_of_memory": True,
                        "device_name": self.gpu_name
                    }
                    results.append
                    break
                else:
                    # Re-raise the error if it's not an out-of-memory error
                    raise e

        output = pd.DataFrame(results)

        # Add more information about the run
        output["context_length"] = self.context_length
        output["batch_size"] = batch_size
        output["compile"] = compile
        output["reduce_overhead"] = reduce_overhead
        output["set_high_precision"] = set_high_precision
        output["device_name"] = self.gpu_name
        if "out_of_memory" not in output.columns:
            output["out_of_memory"] = False

        filename = (
            f"{self.gpu_name.replace(' ', '_')}-batch_size_{batch_size}-compile_{compile}"
            f"-set_high_precision_{set_high_precision}-reduce_overhead_{reduce_overhead}.csv"
        )
        self.save_results(output, filename)

        self.reset()

        if set_high_precision:
            # Go back to default
            torch.set_float32_matmul_precision("highest")
    