from pathlib import Path
from tqdm import tqdm


path = Path("/home/youtian/Documents/pro/pyCode/datasets/TESTASFF/images/temp")
path2 = Path("/home/youtian/Documents/pro/pyCode/datasets/TESTASFF/labels/val")
li = list()
if path.is_dir():
    li = path.rglob("*.jpg")
dst_list = [path2 / i.with_suffix(".txt").name for i in li]
dst_path = Path.cwd() / "test"

if not dst_path.exists():
    Path.mkdir(dst_path)

for i in tqdm(dst_list):
    path3 = dst_path / Path(i.name)
    if not path3.exists():
        path3.touch()
    else:
        raise FileExistsError
    path3.write_text(i.read_text())

