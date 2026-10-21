#!/usr/bin/env bash
# macOS without Homebrew: XGBoost's wheel expects libomp.dylib on its rpath.
# scikit-learn bundles one, so add its directory as an rpath of libxgboost.
set -euo pipefail
PY="${1:-.venv/bin/python}"
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SITE/xgboost/lib/libxgboost.dylib"
OMP_DIR="$SITE/sklearn/.dylibs"
install_name_tool -add_rpath "$OMP_DIR" "$LIB" 2>/dev/null || true
codesign --force --sign - "$LIB"
"$PY" -c "import xgboost; print('xgboost', xgboost.__version__, 'OK')"
