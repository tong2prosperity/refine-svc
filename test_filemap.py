import os
import tempfile
import shutil

from utils.audio_assets import FileMap


def run_tests():
    file_map = FileMap("./examples/reference")
    print(file_map.get_all_files())
    return True

if __name__ == '__main__':
    success = run_tests()
    if success:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 部分测试失败!")
