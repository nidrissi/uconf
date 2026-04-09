import cProfile
import pstats
from pathlib import Path

from sage.all import GF

from uconf import compute_chain_complex, euclidean_unordered_configuration_model

model = euclidean_unordered_configuration_model(GF(2), 2)

profile = cProfile.Profile()
profile.enable()
cc = compute_chain_complex(model.module, degrees=range(-1, 1), weight=4)
profile.disable()

report_path = Path("benchmark_profile.txt")

with report_path.open("w") as report_file:
    report_file.write("cProfile report\n")
    report_file.write("===============\n\n")
    report_file.write(
        "Target: compute_chain_complex(model.module, degrees=range(-1, 1), weight=4)\n"
    )
    report_file.write("Profile sort: cumulative\n")
    report_file.write("Top entries: 20\n\n")
    pstats.Stats(profile, stream=report_file).sort_stats("cumulative").print_stats(20)

print(f"Profile written to {report_path}")
