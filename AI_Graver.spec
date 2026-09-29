# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, get_package_paths


_, mediapipe_path = get_package_paths('mediapipe')
mediapipe_library = Path(mediapipe_path) / 'tasks' / 'c' / 'libmediapipe.dll'
_, cv2_path = get_package_paths('cv2')
face_cascade = Path(cv2_path) / 'data' / 'haarcascade_frontalface_default.xml'


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[(str(mediapipe_library), 'mediapipe/tasks/c'), *collect_dynamic_libs('onnxruntime')],
    datas=[
        ('models/selfie_segmenter.tflite', 'models'),
        ('models/GFPGANv1.4.onnx', 'models'),
        ('models/FSRCNN_x4.pb', 'models'),
        ('models/face_detection_yunet_2023mar.onnx', 'models'),
        (str(face_cascade), 'cv2/data'),
    ],
    hiddenimports=[
        'mediapipe.tasks.python.vision.image_segmenter',
        'onnxruntime',
        'onnxruntime.capi._pybind_state',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI_Graver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI_Graver',
)
