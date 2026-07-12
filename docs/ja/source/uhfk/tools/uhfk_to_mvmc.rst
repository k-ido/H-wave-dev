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

スコープ (v3.1)
^^^^^^^^^^^^^^^

- **Sz-free mixed block (SOC、``enable_spin_orbital = true``)** →
  General-SOC path。``orbitalidx_general.def`` (6 列) を消費し、Sz 非保存の
  ``F[up_i, down_j]`` / ``F[down_i, up_j]`` を出力する。Zeeman、Rashba、
  Dresselhaus、一般 σ_x/σ_y 型 1-body coupling をサポート。
- ``eigen.npz["twist_offset"]`` を ``input.toml`` の canonical
  ``BoundaryCondition`` と照合し、stale 入力/eigen のペアを排除する。
- SOC 下では bridge が H-wave の ``Transfer.dat`` から ``trans.def`` も
  出力する (下記参照)。mVMC の ``vmcdry.out`` は Rashba の s!=t 項を
  保存できない。

スコープ (v3.2)
^^^^^^^^^^^^^^^

- **SOC + APBC** を end-to-end でサポート。``trans.def`` 出力に
  ``BoundaryCondition`` 由来の ``boundary_theta`` を透過させる:
  APBC 方向で境界を跨ぐ行 (``(R_x, R_y, R_z)`` が境界越え) は物理的な
  wrap-phase (正の ``R`` 越えで ``exp(i theta_d)``、負の ``R`` 越えで
  ``exp(-i theta_d)``、APBC 方向は ``theta_d = pi``) を掛ける。
  ベースの SOC 符号規約 (下の「符号規約」参照) と同様、この wrap-phase
  規約も **empirically pinned** であり、
  ``tests/validation/uhfk_mvmc_pairproduct/case_soc_rashba_2d_nosub_apbc``
  で E2E 検証済み (mVMC ⟨H⟩ が H-wave UHF から 0.03% 以内)。H-wave 内部
  規約から first-principles で導出することは UHFk path では clean には
  通らない — 詳細は ``tools/_uhfk_to_mvmc/trans_emit.py`` の module
  docstring を参照。
- **SOC + SubShape > [1, 1, 1]** は v3.4 で延期されていた (Codex v3.4
  Rev.2 finding: dual-A 密度 gate では ship される A/F を検証できず、
  reference A が ``sub_offset`` を落とした別物であったため)。v3.5 では
  この経路を、SHIPPING A を直接検証する gauge-lifted 単一 A 密度チェック
  で復活させた — 下の「Scope (v3.5)」を参照。
- **SOC + APBC + SubShape > [1, 1, 1]** の三重組合せは v3.5 でも引き
  続き延期する (Rev.1 finding 2: E2E fixture 未整備、SOC + APBC と
  SOC + SubShape を合成した phase 経路は独立検証されていない)。
  ``enable_spin_orbital = true`` と antiperiodic ``BoundaryCondition``
  と ``SubShape > [1, 1, 1]`` が同時に指定されると、CLI は dispatch
  前に fail-fast する。
- v3.1 SOC ``trans.def`` の符号規約は引き続き **empirically pinned**。
  以前 ``epsilon_k`` swap と ``H = -sum trans`` の合成から導出すると記載
  していたが、``sc.py`` の ``epsilon_k[orb2, orb1]`` swap は UHFk path
  (``uhfk.py``) では行われないため、この swap 由来の解析的導出は本
  ブリッジには当てはまらないことが判明した。ship される規約は
  ``case_soc_rashba_2d_nosub`` に対する ComplexUHF 検証で 4.4e-8%
  で再検証されており、詳細は ``tools/_uhfk_to_mvmc/trans_emit.py`` の
  module docstring (empirical basis と、放棄された derivation の
  経緯) を参照。

スコープ (v3.5)
^^^^^^^^^^^^^^^

- **SOC + SubShape > [1, 1, 1]** は v3.5 で、gauge-lifted 密度チェック
  ``compare_against_green_sublattice(..., is_soc_sublattice_mode=True)``
  (element-wise 1e-10) を経由してサポートされた。密度チェックは H-wave
  の ``green_sublattice`` (folded-Bloch basis) を ``gauge_lift`` で物理
  基底に持ち上げ、SHIPPING ``conj(A) @ A.T`` (= mVMC に emit する A
  そのもの) と element-wise 照合する。これにより v3.4 の dual-A
  gap を塞ぐ。gauge 変換は shipping A の位相
  ``exp(-i k · (folded_cell + sub_offset))`` を ``green_sublattice``
  の folded-Bloch 保存形式に接続する; 導出は
  ``docs/superpowers/specs/2026-07-05-uhfk-mvmc-pairproduct-general-v35-design.md``
  §2-3 を参照 (``docs/superpowers/`` は gitignored で、リリースには
  spec は同梱されない)。v3.4 の ``_soc_reference_convention``
  escape hatch は削除され、コードベースには shipping 規約の A のみが
  残る。
- **SOC + APBC + SubShape > [1, 1, 1]** の三重組合せは引き続き
  reject される (合成 phase 経路が未検証、E2E fixture 無し — 上の
  「Scope (v3.2)」も参照)。
- SOC + SubShape > [1, 1, 1] で必須の CLI フラグ: ``--transfer``
  (H-wave ``Transfer.dat`` の読み込み)、``--emit-trans`` (Rashba の
  spin-off-diagonal を保持した mVMC ``trans.def`` の出力)、
  ``--emit-orbitalidx`` (StdFace の class merging を bypass。折り
  畳み格子下では Sz 非保存クラスを表現できない — v3.2 spec §1 の
  rationale を参照)。
- Empirical E2E: ``case_soc_rashba_2d_sub`` で mVMC ⟨H⟩ が
  H-wave ``Energy_Total`` と 0.22% 一致 (delta -0.055 / 25.10)、
  gauge-lifted 密度 gate は 1e-10 で clean。

スコープ (v3.6)
^^^^^^^^^^^^^^^

- **SOC + 単方向 APBC + SubShape > [1, 1, 1]** を v3.6 でサポートする。
  ``build_slater_orbitals`` が sub_offset gauge
  ``exp(-i k_folded · (folded_cell + sub_offset))`` と APBC twist
  ``exp(-i theta · r_phys / L_phys)`` を合成し、shipping A に両位相が
  一貫して乗る。``tools/_uhfk_to_mvmc/density_check.py`` の
  ``gauge_lift`` は ``boundary_theta`` を受け取り、同じ合成変換で
  ``green_sublattice`` を物理基底に持ち上げる。shipping 密度 gate
  ``compare_against_green_sublattice(..., is_soc_sublattice_mode=True)``
  は v3.6 の shipping fixture ``case_soc_rashba_2d_sub_apbc``
  (``CellShape = [6, 4, 1]``, ``SubShape = [2, 2, 1]``,
  ``BoundaryCondition = ["antiperiodic", "periodic", "periodic"]``,
  ``enable_spin_orbital = true``, Rashba ``alpha = 0.5``, ``U = 2``,
  ``Ncond = 8``) 上で 1e-10 を通過する。

- **多方向 APBC (n_apbc_dirs >= 2) + SOC + SubShape > [1, 1, 1]**
  は引き続き pre-dispatch で reject する。CLI reject メッセージ::

      ERROR: enable_spin_orbital = true + multi-direction APBC
      (n_apbc_dirs=<count>) + SubShape > [1, 1, 1] is deferred to v3.7.
      Single-direction APBC + SOC + SubShape > 1 is supported in v3.6 as
      of case_soc_rashba_2d_sub_apbc.

  Phase 6 の dispatch 述語::

      n_apbc_dirs = sum(1 for t in theta if abs(t - pi) < 1e-12
                                       or abs(t + pi) < 1e-12)
      if is_soc_mode and n_apbc_dirs > 1 and any(s != 1 for s in sub_shape):
          return 2

  多方向 APBC + SubShape > 1 の合成 twist gauge は第一原理からの
  導出が E2E fixture 上で未検証 (``case_soc_rashba_2d_sub_apbc_apbc``
  相当は v3.6 tree に無い)。誤った ``trans.def`` / shipping A を
  silent に生成することを避けるため fail-fast で reject する。

- **v3.6 七 gate 契約** (fresh workspace ``case_soc_rashba_2d_sub_apbc``
  で全 gate が PASS)::

    1. G0-writer-check: emitted-F の rank-lift ノイズ off 経路が
       aggregate 済 (mapping, params) と 1e-10 で一致
    2. G1: shipping ``build_slater_orbitals`` 密度が gauge_lift-lifted
       ``green_sublattice`` と 1e-10 で一致
    3. G2a-emitted-F: emitted-F projector 密度が ComplexUHF 一体
       Green と 1e-6 で一致
    4. G2a-in-memory-A: in-memory shipping A 密度が ComplexUHF と
       1e-6 で一致
    5. G2b: gauge-lifted ``green_sublattice`` が ComplexUHF と 1e-6 で一致
    6. G3: mVMC ⟨H⟩ と H-wave ``Energy_Total`` の相対 delta ≤ 1 %
    7. G4: ``composite_element.json`` の合成 ``(i_c, s_c, j_c, t_c)``
       が現行 SCF 上で維持され、M-gauge-1..5 + M-ship-1..5 の 10
       mutation 全てが ``T_M = max(1e-5, 0.10 * |G_base|)`` (spec §4.4)
       を超過

  gate は
  ``tests/validation/uhfk_mvmc_pairproduct/run.sh case_soc_rashba_2d_sub_apbc``
  から実行される。各 gate は
  ``^GATE_NAME PASS mode=... `` 形式の anchored PASS 行を出力し、
  ``awk 'index($0, p) == 1'`` で検証する。

- **v3.6 hardening pass (Codex adversarial-review 2026-07-12)**。
  Phase 6 の初回 seven-gate PASS 後、4 ラウンドの adversarial review で
  一連の findings が浮上し、以下のように修正した:

  * **G1 / G2a / G2b 実比較**: 初回の dispatcher は ``max_abs_delta=0.0``
    で PASS を印字する stub で、実際の numeric 比較は行っていなかった。
    ``compare.py`` を全面書き換えし、workspace の H-wave outputs を読み
    込んで ``build_slater_orbitals`` で shipping A を組み、
    ``gauge_lift`` (workspace の ``boundary_theta``) で
    ``green_sublattice`` を持ち上げ、strict-parse された ComplexUHF
    ``zvo_UHF_cisajs.dat`` と比較する形にした。ファイル欠損 / 打切り /
    形式不正 / 重複 / 範囲外 / 非有限値は新設 ``ComplexUHFParseError``
    で fail closed。

  * **ComplexUHF Cross-solver seeding**: Rashba + APBC 下で H-wave の
    broken-symmetry 極小と ComplexUHF の random-init default は異なり、
    2 つの独立 SCF が別の valid 極小に落ちて 4.76e-2 の要素毎密度差が
    生じた。修正: ``scripts/seed_complexuhf_from_hwave.py`` が
    H-wave の shipping A 密度を ComplexUHF ``initial.def`` (
    ``IgnoreLinesInDef=5`` 規約に沿った 5 行 header) に書き出し、微小な
    Hermitian ``perturb-scale`` (このフィクスチャでは 1e-6) を付与
    することで ComplexUHF を実際に反復させる (``run.sh`` が
    ``uhf.log`` の ``finished at N step`` を awk parse し N >= 1 を強制)
    が H-wave の basin を離脱しない範囲に留める。seeding 後の
    ComplexUHF は 9 SCF steps 走り ``Energy_Total = -25.3717166``
    (H-wave と 10 桁一致) に収束する。

  * **Snapshot workspace rejection**: snapshot guard が ``tests/data``
    をプロセス CWD 基準で resolve していたため、リポジトリ外から起動
    すると check が silent に bypass されていた。修正: guard module
    自身の場所 (``Path(__file__).resolve().parents[3]``) に anchor し、
    anchored root が存在しなければ fail closed。

  * **G4 shadow-copy drift**: topology guard の
    ``_build_A_ship_mutated`` が SOC branch を inline に再実装しつつ、
    baseline を canonical kernel と比較していなかった。
    ``build_slater_orbitals`` に drift が入っても G4 が見逃す可能性が
    あった。修正: mutation matrix 実行前に canonical kernel との 1e-10
    等値 assert を必須化。

  * **Loader-injection env sanitization**: ``run.sh`` が caller 側の
    ``LD_LIBRARY_PATH`` / ``LD_PRELOAD`` / ``LD_AUDIT`` を子プロセス
    (``vmcdry.out`` / ``vmc.out`` / ``UHF``) にそのまま継承していた。
    修正: (a) 上記変数 + ``BASH_ENV`` / ``ENV`` を top-level で
    ``unset``, (b) ``trap - DEBUG ERR RETURN EXIT`` で継承 trap を
    クリア, (c) ``run_native()`` wrapper が
    ``env -u LD_PRELOAD -u LD_AUDIT [-u LD_LIBRARY_PATH|
    LD_LIBRARY_PATH=<validated>]`` を用いて各 native solver 呼び出し
    毎に per-command 消毒, (d) ``MVMC_LD_LIBRARY_PATH`` を絶対パス /
    現在の user 所有 / non-world-writable / 単一ディレクトリで validate。

  各 findings に対応する regression tests を追加: loader-env
  sanitization に 5 tests (`tests/test_run_sh_loader_env_sanitize.py`)、
  ComplexUHF strict parser に 6 tests
  (`tests/test_uhfk_mvmc_pairproduct_compare_wiring.py`)、
  CWD-independent snapshot guard に 3 tests
  (`tests/test_snapshot_rejection_guard_v36.py`)。

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

Dispatch (v3.1、6-case)
^^^^^^^^^^^^^^^^^^^^^^^

CLI は ``input.toml`` を parse した後
``is_soc_mode = toml_param.get("enable_spin_orbital", False)`` を計算し、
``(is_antiparallel_metadata, orbitalidx_format, is_soc_mode)`` の組で
分岐する:

- ``(True, antiparallel, False)`` — v2.1 AntiParallel (変更なし)。
- ``(True, general, False)`` — v3 forced-General。
- ``(False, general, False)`` — v3 General (A/B)。
- ``(False, antiparallel, False)`` — 拒絶。
- ``(\*, general, True)`` — **v3.1 General-SOC**。
- ``(\*, antiparallel, True)`` — 拒絶 (SOC は 6 列必須)。

BoundaryCondition の契約 (v3.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

bridge は ``BoundaryCondition`` を H-wave の共有 helper
``normalize_boundary_condition`` に委譲して正規化する。受理する形式:

- PBC: ``"p"``, ``"periodic"`` (case-insensitive、whitespace-stripped)
- APBC: ``"ap"``, ``"antiperiodic"`` (同上)

これ以外の文字列は dispatch 前に ``ValueError``。raw fallback は存在しない。
key を省略すると all-PBC を default とする。

Bridge trans.def 出力 (v3.1 SOC 限定)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SOC 下では mVMC の ``vmcdry.out`` が ``StdFace_Hopping`` で ``trans.def``
を作るが、spin-diagonal のみで Rashba の s != t 項が落ちる。この gap を
埋めるため、bridge が H-wave の ``Transfer.dat`` (Wannier90 風形式、
``iWan = 2 * a_phys + spin + 1``) を読み、mVMC の ``trans.def`` を
``(i, s, j, t, re, im)`` 形式で書き出す。Rashba の spin-off-diagonal を
保持する。

符号規約 (実験的に固定):

- ``s == t`` (NN hopping): ``trans = -val_hwave`` (vmcdry の flip と一致)。
- ``s != t`` (Rashba): ``trans = +val_hwave``。

この mixed 規約 ``(sign_diag = -1, sign_offdiag = +1)`` は
``case_soc_rashba_2d_nosub`` に対する ComplexUHF 検証で 4.4e-8% 一致
という形で **empirically pinned** されている。mVMC の規約は
``H = -Σ trans c†c`` だが、source-target index 規約と H-wave の k-空間
Hamiltonian との相互作用の詳細は、本ブリッジ内では **first principles
のみからは導出できない** — 詳細および放棄された「H-wave ``sc.py`` の
swap から導出する」story は ``tools/_uhfk_to_mvmc/trans_emit.py`` の
module docstring を参照 (``sc.py`` は ``epsilon_k[orb2, orb1]`` swap を
行うが、``uhfk.py`` は同じ swap を行わないため、swap 由来の導出は
合成できない)。一様 flip (両方 ``-val``) では同じ fixture で ⟨H⟩ が
42.58% ずれる。**mVMC または H-wave のバージョン更新後は
``case_soc_rashba_2d_nosub`` に対する E2E で再検証する** — mVMC 側の
機構は ``locgrn_fsz.c:128`` に mVMC 作者の ``//TBC`` コメントが残る。

新規 CLI フラグ (SOC 限定): ``--transfer <path>`` と ``--emit-trans <path>``。

v3.5 のスコープ外
^^^^^^^^^^^^^^^^^

- SOC + APBC + ``SubShape > [1, 1, 1]`` の三重組合せは v3.5 以降へ
  延期 (Rev.1 finding 2: E2E fixture 未整備、SOC + APBC と
  SOC + SubShape を合成した phase 経路は独立検証されていない)。
  2 組の subset は fixture を保持している:
  ``case_soc_rashba_2d_nosub_apbc`` (SOC + APBC、0.03% 差)、
  ``case_soc_rashba_2d_sub`` (SOC + SubShape、0.22% 差)。
- 2-body Sz 非保存相互作用 (spin-flip Coulomb, Hund coupling,
  pair hopping)。CoulombIntra (on-site U) のみが v3.5 の唯一の 2-body 項。

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

SOC + SubShape > [1, 1, 1] では密度チェックは
``compare_against_green_sublattice(..., is_soc_sublattice_mode=True)``
(許容 1e-10) に切り替わる (H-wave ``greenone.dat`` の fold path は
本組合せで known-buggy、``green_sublattice`` が source of truth)。
チェックは H-wave の ``green_sublattice`` を ``gauge_lift`` で物理
基底に持ち上げ、SHIPPING ``conj(A) @ A.T`` (mVMC に emit する A
そのもの) と element-wise 照合するため、shipping-A 経路の regression
は silent には通らない。これにより v3.4 で塞いだ密度 gate (dual-A
hole) が v3.5 spec §2-3 で導出した gauge lift により復活し、
v3.4 の ``_soc_reference_convention`` escape hatch は削除された。

終了コード
^^^^^^^^^^

- ``0`` — 成功
- ``2`` — fail-fast ガードが入力を拒絶 (スコープ外モード、Sz-free SCF、
  有限温度の分数占有残留、``orbitalidx.def`` / geometry / 境界条件の
  不整合など)。stderr に失敗した検査名を明示する。
- ``3`` — ``--check-density`` が許容差を超える不一致を検出
