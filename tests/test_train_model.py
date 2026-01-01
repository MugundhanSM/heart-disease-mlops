import subprocess
import sys

def test_train_script_runs():
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "Training script failed"
    assert "Training completed successfully" in result.stdout, "Training did not complete successfully"