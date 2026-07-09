.. highlight:: none

uhfk_to_mvmc.py — UHFk → mVMC PairProduct ブリッジ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tools/uhfk_to_mvmc.py`` は H-wave UHFk の SCF 結果を、mVMC の
``InOrbital`` / ``InOrbitalAntiParallel`` 初期波動関数ファイル
(``zqp_orbital_uhfk.dat``) に変換するスクリプトである。これにより
mVMC の PairProduct 状態を H-wave UHF の Slater 行列で初期化できる。

スコープ (v2):

- 単軌道 (``norb_orig = 1``)。副格子フォールドは ``SubShape`` が各方向で
  ``CellShape`` を割り切る任意の値に対応。``SubShape`` 省略時は
  ``CellShape`` にフォールバック (``uhfk.py:_init_lattice`` と一致)。
- Sz-fixed UHF のみ (``2Sz = 0`` で ``N_up = N_down``)
- PBC・APBC とも決定論的にサポート。ブリッジは、H-wave の負号ゲージ
  APBC 変換 (``tilde_c_r = exp(-i theta r / L_phys) c_r``) と、
  ``ifftn(..., norm='forward')`` に対応する tilde 側の正号 Bloch
  Fourier (``c_R = (1/sqrt(N_folded)) sum_k c_k exp(+i k R)``) から
  導かれる **正号 Bloch 振幅規約** を採用する。
  ``SubShape = [1, 1, 1]`` では (k, -k) 時間反転対称和で plane wave 因子
  が対称化されるため密度は符号選択に依存しないが、``SubShape > [1, 1, 1]``
  では折り畳み固有ベクトルの副格子 envelope が非自明になるため、H-wave
  ``greenone.dat`` と element-wise 一致するのは本規約のみ (SubShape=[2,1,1]
  APBC L=8 fixture で 1e-14 に確認済み)。
- (k, -k) 時間反転対 AntiParallel のみ。磁性 / spin 依存占有で
  ``(k_row, local_band)`` pair-closure が満たされない場合は
  ``build_amplitudes`` 入口で明示的に拒絶する。
- T=0 への Slater 投影。Fermi 準位付近に分数占有が残る有限温度 SCF は拒絶
  (より小さい ``T`` で再計算するよう促す)
- ``params[idx]`` に微小な一様乱数(default 振幅 ``1e-8``、
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
