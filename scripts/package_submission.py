"""Create a submission-ready artifact bundle under results/submission_package."""

import hashlib
import os
import shutil
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
PACKAGE_DIR = os.path.join(RESULTS_DIR, "submission_package")
ZIP_BASENAME = os.path.join(RESULTS_DIR, "submission_package")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src_rel, dst_rel):
    src = os.path.join(ROOT, src_rel)
    dst = os.path.join(PACKAGE_DIR, dst_rel)
    if not os.path.isfile(src):
        return False, src_rel
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True, src_rel


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.isdir(PACKAGE_DIR):
        shutil.rmtree(PACKAGE_DIR)
    os.makedirs(PACKAGE_DIR, exist_ok=True)

    copy_plan = [
        ("README.md", "README.md"),
        ("requirements.txt", "requirements.txt"),
        ("run.sh", "run.sh"),
        ("results/full_report.md", "report/full_report.md"),
        ("results/main_dashboard.png", "figures/main_dashboard.png"),
        ("results/clarification_quality_entropy_delta.png", "figures/clarification_quality_entropy_delta.png"),
        ("results/ablations_dashboard.png", "figures/ablations_dashboard.png"),
        ("results/metrics.csv", "data/metrics.csv"),
        ("results/metrics_ablations.csv", "data/metrics_ablations.csv"),
        ("results/metrics_robust_answer_noise.csv", "data/metrics_robust_answer_noise.csv"),
        ("results/metrics_robust_mismatch.csv", "data/metrics_robust_mismatch.csv"),
        ("results/milestone_feb13_feb20_report.md", "report/milestone_feb13_feb20_report.md"),
        ("results/robust_answer_noise_deltas.png", "figures/robust_answer_noise_deltas.png"),
        ("results/robust_mismatch_deltas.png", "figures/robust_mismatch_deltas.png"),
        ("docs/presentation_slides.md", "presentation/presentation_slides.md"),
        ("docs/presentation_talk_track.md", "presentation/presentation_talk_track.md"),
        ("docs/executive_summary.md", "presentation/executive_summary.md"),
        ("docs/final_report.md", "report/final_report.md"),
        ("docs/submission_checklist.md", "presentation/submission_checklist.md"),
    ]

    copied = []
    missing = []
    for src_rel, dst_rel in copy_plan:
        ok, rel = _copy(src_rel, dst_rel)
        if ok:
            copied.append((src_rel, dst_rel))
        else:
            missing.append(rel)

    reproduce = os.path.join(PACKAGE_DIR, "REPRODUCE.txt")
    with open(reproduce, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "AskOrAct reproducibility commands (Linux/macOS):",
                    "",
                    "./run.sh setup",
                    "./run.sh sweep",
                    "./run.sh ablations",
                    "./run.sh report",
                    "./run.sh package",
                    "",
                    "Outputs:",
                    "results/metrics.csv",
                    "results/metrics_ablations.csv",
                    "results/full_report.md",
                    "results/main_dashboard.png",
                    "results/ablations_dashboard.png",
                ]
            )
        )

    manifest = os.path.join(PACKAGE_DIR, "MANIFEST.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Bundle: submission_package\n\n")
        f.write("Included files:\n")
        for _src_rel, dst_rel in copied:
            out_path = os.path.join(PACKAGE_DIR, dst_rel)
            size = os.path.getsize(out_path)
            sha = _sha256(out_path)
            f.write(f"- {dst_rel} | {size} bytes | sha256={sha}\n")
        if missing:
            f.write("\nMissing source files (not copied):\n")
            for rel in missing:
                f.write(f"- {rel}\n")

    if os.path.isfile(ZIP_BASENAME + ".zip"):
        os.remove(ZIP_BASENAME + ".zip")
    shutil.make_archive(ZIP_BASENAME, "zip", PACKAGE_DIR)

    print(f"Wrote {PACKAGE_DIR}")
    print(f"Wrote {ZIP_BASENAME}.zip")
    if missing:
        print("Warning: some expected files were missing:")
        for rel in missing:
            print(" -", rel)


if __name__ == "__main__":
    main()
