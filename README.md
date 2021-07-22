# Proc_CEOS
"Proc_CEOS"はALOS2 L1.1 CEOSフォーマットの処理と作成した学習データを用いてセマンティックセグメンテーションを行います．

# できること
- CEOS_module
    - 強度画像，位相画像の作成
    - 緯度，経度の計算
    - JAXA提供の土地被覆図からGT画像の作成
    - Pauli分解画像の作成

# 必要パッケージ
- Python
    - opencv-python
    - scipy
    - numpy
    - joblib
    - matplotlib
    - Pillow