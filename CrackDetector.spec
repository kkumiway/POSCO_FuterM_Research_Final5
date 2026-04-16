# CrackDetector.spec
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
block_cipher = None

from PyInstaller.utils.hooks import collect_all
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
xgb_datas, xgb_binaries, xgb_hiddenimports = collect_all('xgboost')  # ← 추가

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[*numpy_binaries, *scipy_binaries, *xgb_binaries],
    datas=[
        ('Mainwindow.ui', '.'),
        *numpy_datas,
        *scipy_datas,
        *xgb_datas,
    ],
    hiddenimports=[
        *numpy_hiddenimports,
        *scipy_hiddenimports,
        *xgb_hiddenimports,
        'sklearn', 'sklearn.ensemble', 'sklearn.svm', 'sklearn.linear_model',
        'sklearn.neural_network', 'sklearn.preprocessing', 'sklearn.pipeline',
        'sklearn.model_selection', 'sklearn.metrics', 'sklearn.utils._cython_blas',
        'sklearn.neighbors._partition_nodes',
        'xgboost',
        'torch', 'torch.nn', 'torch.optim', 'torch.utils.data',
        'torchvision', 'torchvision.models', 'torchvision.transforms',
        'scipy', 'scipy.signal', 'scipy.fft', 'scipy.stats', 'scipy.integrate',
        'matplotlib', 'matplotlib.backends.backend_qt5agg',
        'numpy', 'pandas',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tensorflow', 'tensorflow_core', 'tensorboard', 'keras'],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CrackDetector',
    debug=False,
    strip=False,
    upx=False,
    console=False,
    cipher=block_cipher,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='CrackDetector',
)