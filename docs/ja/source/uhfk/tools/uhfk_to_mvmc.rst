.. highlight:: none

uhfk_to_mvmc.py — UHFk → mVMC PairProduct ブリッジ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tools/uhfk_to_mvmc.py`` は H-wave UHFk の SCF 結果を、mVMC の
``InOrbital`` / ``InOrbitalAntiParallel`` 初期波動関数ファイル
(``zqp_orbital_uhfk.dat``) に変換するスクリプトである。これにより
mVMC の PairProduct 状態を H-wave UHF の Slater 行列で初期化できる。

スコープ (v1):

- 単軌道 (``norb_orig = 1``, ``SubShape = [1, 1, 1]``)
- Sz-fixed UHF のみ (``2Sz = 0`` で ``N_up = N_down``)
- PBC・APBC とも決定論的にサポート。ブリッジは負号 Bloch 規約
  (``exp(-i k r)``) を採用し、H-wave の
  ``np.fft.fftn(norm='forward')`` の k→r カーネルと整合する
  ため、両境界条件下で per-spin density が一致する。
- AntiParallel pair form のみ。磁性 / 非対称占有で (k, -k) pair-closure
  が満たされない場合は明示的に拒絶する。
- T=0 への Slater 投影。Fermi 準位付近に分数占有が残る有限温度 SCF は拒絶
  (より小さい ``T`` で再計算するよう促す)
- ``params[idx]`` に微小な一様乱数(default 振幅 ``1e-6``、
  ``--epsilon-noise`` で変更可)を加算して rank-deficient な F に対する
  mVMC の Pfaffian Slater 評価の特異性を回避する。``ComplexType 1``
  では実部・虚部ともに、``ComplexType 0`` では実部のみに乱数を加える。
  mVMC 本体の ComplexUHF と同じ手法
  (``mVMC-1.4.0/src/ComplexUHF/output.c:274``)。

ワークフロー
^^^^^^^^^^^^

1. StdFace で mVMC 入力一式 (``orbitalidx.def`` 等) を生成する。
   ``CalcMode = 2`` と、副格子並進対称性に応じた ``Lsub`` を設定。
   APBC は ``phase0 = 180.0`` を指定。
2. H-wave UHFk SCF を実行し、``eigen.npz`` に加えて新規 ``occupation.npz``
   を ``[file.output]`` で要求する::

       [file.output]
         path_to_output = "output"
         eigen          = "eigen.npz"
         green          = "green.npz"
         occupation     = "occupation.npz"
         onebodyg       = "greenone.dat"

3. ブリッジを実行::

       python tools/uhfk_to_mvmc.py \
           --input        input.toml \
           --eigen        output/eigen.npz \
           --occupation   output/occupation.npz \
           --geometry     geometry_uhf.dat \
           --orbitalidx   mvmc_inputs/orbitalidx.def \
           --output       mvmc_inputs/zqp_orbital_uhfk.dat \
           --check-density \
           --onebodyg-uhf output/greenone.dat

4. mVMC の ``namelist.def`` に ``InOrbital zqp_orbital_uhfk.dat``
   (または ``InOrbitalAntiParallel zqp_orbital_uhfk.dat``) を追加すれば、
   mVMC が PairProduct パラメータを本ファイルで初期化する。

密度行列チェック (推奨)
^^^^^^^^^^^^^^^^^^^^^^^

``--check-density`` は (k, -k) pair 構築から 1 体密度行列を構築し、
H-wave の物理基底 ``greenone.dat`` と element-wise 比較する (許容 1e-10)。
不一致は fatal で、H-wave APBC、ブリッジ、もしくは geometry の前提に
バグがあることを示す。

終了コード
^^^^^^^^^^

- ``0`` — 成功
- ``2`` — fail-fast ガードが入力を拒絶 (スコープ外モード、Sz-free SCF、
  有限温度の分数占有残留、``orbitalidx.def`` / geometry / 境界条件の
  不整合など)。stderr に失敗した検査名を明示する。
- ``3`` — ``--check-density`` が許容差を超える不一致を検出
