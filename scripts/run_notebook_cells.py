from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from scipy.io import loadmat


DATA_FILE = "voc_dataset_table1_1+2_vs_3_bpd.mat"


def validate_data(repo_root: Path) -> Path:
    data_path = repo_root / "data" / DATA_FILE
    if not data_path.is_file():
        raise FileNotFoundError(
            f"缺少私有数据：{data_path}\n"
            "请按 data/README.md 申请并放置数据。"
        )

    data = loadmat(data_path)
    missing = {"X", "y"} - data.keys()
    if missing:
        raise ValueError(f"MAT 文件缺少字段：{sorted(missing)}")
    if data["X"].shape != (159, 1692):
        raise ValueError(f"X 形状应为 (159, 1692)，实际为 {data['X'].shape}")
    if data["y"].reshape(-1).shape != (159,):
        raise ValueError(f"y 应包含 159 个标签，实际为 {data['y'].shape}")
    return data_path


def execute_cells(
    *,
    repo_root: Path,
    notebook_path: Path,
    cell_indices: list[int],
    output_path: Path,
    import_paths: list[Path],
    module_override: Path | None = None,
    check_only: bool = False,
) -> None:
    repo_root = repo_root.resolve()
    notebook_path = notebook_path.resolve()
    validate_data(repo_root)

    notebook = nbformat.read(notebook_path, as_version=4)
    invalid = [index for index in cell_indices if index >= len(notebook.cells)]
    if invalid:
        raise IndexError(f"Notebook 不存在单元：{invalid}")
    if any(notebook.cells[index].cell_type != "code" for index in cell_indices):
        raise ValueError("所选 Notebook 单元必须全部为代码单元。")

    if check_only:
        print(f"数据检查通过：{repo_root / 'data' / DATA_FILE}")
        print(f"Notebook 检查通过：{notebook_path}")
        print(f"将执行代码单元：{cell_indices}")
        return

    with tempfile.TemporaryDirectory(prefix="bpd-repro-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        kernel_paths = [Path(path).resolve() for path in import_paths]

        if module_override is not None:
            shutil.copy2(module_override, temp_dir / "modules.py")
            kernel_paths.insert(0, temp_dir)

        path_literals = ", ".join(repr(str(path)) for path in kernel_paths)
        bootstrap = nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os, sys\n"
            f"PROJECT_ROOT = Path({str(repo_root)!r})\n"
            "os.chdir(PROJECT_ROOT)\n"
            f"for _path in reversed([{path_literals}]):\n"
            "    if _path not in sys.path:\n"
            "        sys.path.insert(0, _path)\n"
        )

        selected = [bootstrap]
        for index in cell_indices:
            cell = copy.deepcopy(notebook.cells[index])
            cell.execution_count = None
            cell.outputs = []
            selected.append(cell)

        executable = nbformat.v4.new_notebook(
            cells=selected,
            metadata=copy.deepcopy(notebook.metadata),
        )
        executable.metadata.setdefault("kernelspec", {})["name"] = "python3"

        client = NotebookClient(
            executable,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(repo_root)}},
        )
        client.execute()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executable, output_path)
    print(f"复现完成，执行记录已保存：{output_path}")
