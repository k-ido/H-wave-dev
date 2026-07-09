.. highlight:: none

uhfk_to_mvmc.py — UHFk → mVMC PairProduct ブリッジ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tools/uhfk_to_mvmc.py`` は H-wave UHFk の SCF 結果を、mVMC の
``InOrbital`` / ``InOrbitalAntiParallel`` 初期波動関数ファイル
(``zqp_orbital_uhfk.dat``) に変換するスクリプトである。これにより
mVMC の PairProduct 状態を H-wave UHF の Slater 行列で初期化できる。

スコープ (v3):

- 単軌道 (``norb_orig = 1``)。副格子フォールドは ``SubShape`` が各方向で
  ``CellShape`` を割り切る任意の値に対応。``SubShape`` 省略時は
  ``CellShape`` にフォールバック (``uhfk.py:_init_lattice`` と一致)。
- **Sz-fixed UHF (2Sz = 0)** → AntiParallel path (v1/v2/v2.1 と同じ
  挙動、``orbitalidx.def`` の 3 or 4 列形式を消費)。
- **Sz-fixed 2Sz ≠ 0 (A ケース) と Sz-free 非 mixed (B ケース)** →
  General path (v3、``orbitalidx_general.def`` の 6 列形式を消費)。
  CLI は ``(is_antiparallel_metadata, orbitalidx_format)`` の組で
  自動 dispatch する。ユーザは StdFace で ``orbitalidx_general.def``
  を生成する必要がある (A ケースでは ``stan.in`` の ``2Sz`` を非 0 に
  設定する)。
- スピン不均衡 Slater については、canonical ``(k, partner(k))`` ブロック
  での same-spin excess pair 発行により **同スピン pair 成分**
  (``F[up, up]``, ``F[down, down]``) をサポートする。
- Sz-free で mixed block を含む状態 (``column_spin = -1``、SOC /
  spin-orbital mode) は **v3 スコープ外**、v3.1 で扱う。
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
- AntiParallel path: (k, -k) 時間反転対、``(k_row, local_band)`` 上で
  pair-closure。General path (v3): canonical ``(k, partner(k))``
  ブロック内で cross + same-spin excess pair を発行。§3.2 の
  pair-closure 条件 (canonical block の same-spin excess 不整合、
  self-pair の奇数 excess など) を満たさない占有は
  ``validate_general_prerequisites`` で明示的に拒絶する。
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
   (または ``InOrbitalAntiParallel zqp_orbital_uhfk.dat``、v3 General
   path なら ``InOrbitalGeneral zqp_orbital_uhfk.dat``) を追加すれば、
   mVMC が PairProduct パラメータを本ファイルで初期化する。

Dispatch (v3)
^^^^^^^^^^^^^

CLI は ``--orbitalidx`` を先に parse し、
``(is_antiparallel_metadata, orbitalidx_format)`` の組で経路を選択する:

- ``(True, antiparallel)`` — v2.1 AntiParallel path (v1/v2/v2.1 の
  挙動そのまま)。
- ``(True, general)`` — forced-General 分岐。占有集合が v2.1 の
  ``(k_row, local_band)`` pair-closure を満たしていれば
  ``F[up, down]`` は 1e-12 で v2.1 の F を再現する。満たしていない
  (spin canting 由来の up-up excess 等) 場合は WARNING を出しつつ、
  出力は正しい ``InOrbitalGeneral`` state となる (ただし v2.1 経路
  では再現不可)。
- ``(False, general)`` — v3 General path (A + B スコープ)。
- ``(False, antiparallel)`` — 拒絶。StdFace を非 0 の ``2Sz`` (A) や
  Zeeman 駆動 Sz-free (B) 設定で回して ``orbitalidx_general.def`` を
  再生成すること。

``is_antiparallel_metadata`` は ``input.toml`` の ``2Sz`` が明示的に 0、
``N_up == N_down``、``column_spin ∈ {0, 1}``、``column_mu_group`` の
unique 数が 2、column_spin と mu_group が bijective であることを **全て**
満たしたときのみ True になる。

クラス一致性チェック (v3 General)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

General path では ``aggregate_general_orbital_params`` が、
平均化する **前** に ``orbitalidx_general.def`` の各クラスに割り当てられた
sign 付き F 成分が ``class_consistency_tol`` (default 1e-8) 以内で
一致していることを検査する。不一致であれば
``ClassInconsistencyError`` を offending idx と観測された max residual
とともに raise する。これにより、StdFace が生成したクラスが仮定する
対称性を Slater state が守っていないケース (対称なハミルトニアンで
自発的対称性破れの UHF 基底状態が出るなど) を silent averaging から
守る。

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
